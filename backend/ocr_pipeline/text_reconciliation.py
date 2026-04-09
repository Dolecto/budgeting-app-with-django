"""
ocr_pipeline/text_reconcilation.py
==================================
Reconcile multiple OCR pipeline JSON outputs from text_extraction 
for a single receipt image, then extract structured data based on receipt type.
"""

import json, re
from collections import defaultdict


# Structural patterns
PRICE_RE = re.compile(r'^-?\d+[.,]\d{1,3}$')       # matches: 2.59  -0.30  20,08  4.47  1,83
VATCODE_RE = re.compile(r'^([A-Z])\1{0,7}$')       # VAT-code column artefacts
SHORT_CAP_PREFIX = re.compile(r'^[A-ZÄÖÜ]{2,3}$')  # store-internal category codes


# Geometry and number helpers
def cx(box): return (box[0]+box[2])/2.0
def cy(box): return (box[1]+box[3])/2.0
def bh(box): return box[3]-box[1]


def parse_number(text):
    """Parse a price string to float."""
    c = text.strip().replace(" ", "")
    # European decimal: 20,08 → 20.08
    if re.match(r'^-?\d+,\d{1,3}$', c):
        c = c.replace(",", ".")
    # 1.234,56 → 1234.56
    elif re.match(r'^-?\d{1,3}(\.\d{3})+,\d{2}$', c):
        c = c.replace(".", "").replace(",", ".")
    # 1,234.56 → 1234.56
    elif re.match(r'^-?\d{1,3}(,\d{3})+\.\d{2}$', c):
        c = c.replace(",", "")
    try:    return float(c)
    except: return None


def is_price(text):
    return bool(PRICE_RE.match(text.strip()))


def is_noise(text):
    """True for single-char tokens or VAT-code column artefacts."""
    t = text.strip()
    if len(t) <= 1:
        return True
    if VATCODE_RE.match(t):
        return True
    return False


def cents(value):
    """Convert float price to integer cents for exact comparison."""
    return round(value * 100)


# Reconciliation pipeline
# Step 1 - Load one pipeline JSON
def load_pipeline(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tokens = []
    for text, score, box in zip(
            data.get("rec_texts", []),
            data.get("rec_scores", []),
            data.get("rec_boxes", [])):
        text = text.strip()
        if text and score >= 0.15:
            tokens.append({"text": text, "score": score, "box": box})
    return tokens


# Step 2 - Per-pipeline: group into rows, detect price column, parse rows
def _group_into_rows(tokens):
    if not tokens: return []
    tokens = sorted(tokens, key=lambda t: cy(t["box"]))
    heights = sorted([bh(t["box"]) for t in tokens])
    tol = heights[len(heights) // 2] * 0.55
    rows = [[tokens[0]]]; cur = cy(tokens[0]["box"])
    for tok in tokens[1:]:
        tc = cy(tok["box"])
        if abs(tc - cur) <= tol: rows[-1].append(tok)
        else: rows.append([tok]); cur = tc
    return [sorted(r, key=lambda t: t["box"][0]) for r in rows]


def _detect_price_col(tokens):
    """
    The price column is the rightmost cluster of strict-price tokens.
    We take the top-right 45% of all price x-centres and build a band.
    No tolerance numbers: 'top 45%' is a rank, not a pixel distance.
    """
    price_xs = [cx(t["box"]) for t in tokens if is_price(t["text"])]
    if not price_xs: return (0.0, float("inf"))
    price_xs.sort()
    cut    = len(price_xs) * 55 // 100   # integer rank split
    col_xs = price_xs[cut:] or price_xs
    span   = max(col_xs) - min(col_xs)
    margin = max(40, span // 2 + 30)     # derived from data spread, not a fixed tolerance
    return (min(col_xs) - margin, max(col_xs) + margin)


def _in_col(tok, col): return col[0] <= cx(tok["box"]) <= col[1]
def _is_price_tok(tok, col): return is_price(tok["text"]) and _in_col(tok, col)


def parse_pipeline_rows(tokens):
    col  = _detect_price_col(tokens)
    rows = _group_into_rows(tokens)
    parsed = []
    for row in rows:
        ptoks = sorted([t for t in row if _is_price_tok(t, col)],
                       key=lambda t: cx(t["box"]), reverse=True)
        ntoks = [t for t in row if not _in_col(t, col) and not is_noise(t["text"])]

        price_raw   = ptoks[0]["text"]  if ptoks else None
        price_val   = parse_number(price_raw) if price_raw else None
        price_score = ptoks[0]["score"] if ptoks else 0.0
        name        = " ".join(t["text"] for t in ntoks).strip()
        row_str     = " ".join(t["text"] for t in row)

        parsed.append({
            "name":        name,
            "price_raw":   price_raw,
            "price":       price_val,
            "price_score": price_score,
            "row_str":     row_str,
            "_name_score": sum(t["score"] for t in ntoks),
        })
    return parsed


# Step 3 - Reconcile: align by price magnitude, weighted vote
def _price_sig(row):
    """Absolute-value cents integer for alignment comparison."""
    v = row["price"]
    return cents(abs(v)) if v is not None else None


def _align_rows(ref_rows, other_rows):
    """
    Greedy forward scan: match other_rows to ref_rows using exact cent match
    on absolute price value.  No-price rows are matched by proximity (small offset).
    """
    n = len(ref_rows); m = len(other_rows)
    aligned = [None] * n; j = 0
    for i in range(n):
        rp = _price_sig(ref_rows[i])
        best_j = None; best_dist = float("inf")
        for dj in range(-2, 4):
            jj = j + dj
            if not (0 <= jj < m): continue
            op = _price_sig(other_rows[jj])
            if rp is not None and op is not None and rp == op:   # exact cent match
                best_j = jj; best_dist = 0; break
            if rp is None and op is None:
                dist = abs(dj)
                if dist < best_dist: best_dist = dist; best_j = jj
        if best_j is not None:
            aligned[i] = other_rows[best_j]; j = best_j + 1
    return aligned


def _wvote(candidates):
    """Return text with highest total weighted score."""
    vote = defaultdict(float)
    for text, w in candidates: vote[text] += w
    return max(vote, key=lambda k: (vote[k], -len(k))) if vote else ""


def reconcile_rows(pipeline_row_lists):
    if not pipeline_row_lists: return []
    ref_rows = max(pipeline_row_lists, key=len)
    n = len(ref_rows)

    name_votes  = [defaultdict(float) for _ in range(n)]
    price_votes = [defaultdict(float) for _ in range(n)]

    for pl_rows in pipeline_row_lists:
        aligned = ref_rows if pl_rows is ref_rows else _align_rows(ref_rows, pl_rows)
        for i, row in enumerate(aligned):
            if row is None: row = ref_rows[i]
            if row["name"]:
                name_votes[i][row["name"]] += row["_name_score"] + 0.01
            if row["price_raw"]:
                price_votes[i][row["price_raw"]] += row["price_score"] + 0.01

    out = []
    for i, ref in enumerate(ref_rows):
        best_name      = _wvote(list(name_votes[i].items()))
        best_price_raw = _wvote(list(price_votes[i].items())) if price_votes[i] else None
        best_price     = parse_number(best_price_raw) if best_price_raw else None
        out.append({
            "name":      best_name,
            "price_raw": best_price_raw,
            "price":     best_price,
            "row_str":   ref["row_str"],
        })
    return out


# Step 3b - Merge orphan name-only rows with the following priced row
def merge_orphan_name_rows(rows):
    """
    If row[i] has a name but no price, and row[i+1] has a price but a
    very short name (≤1 word), merge them: name from row[i], price from row[i+1].
    """
    merged = []; i = 0
    while i < len(rows):
        row = rows[i]
        if (row["name"] and row["price"] is None and i + 1 < len(rows)):
            nxt = rows[i + 1]
            if nxt["price"] is not None and len((nxt["name"] or "").split()) <= 1:
                combined = dict(nxt)
                combined["name"] = row["name"]
                merged.append(combined); i += 2; continue
        merged.append(row); i += 1
    return merged


# Step 4 - Locate total
def find_total_by_running_sum(price_sequence):
    """
    Given a list of (index, price_float) tuples, return the index of the
    first price that exactly equals the running sum of all preceding prices.
    Returns None if no such price is found.
    """
    running_cents = 0
    for idx, price in price_sequence:
        p_cents = cents(price)
        if p_cents == running_cents:
            return idx          # this row IS the total
        running_cents += p_cents
    return None


# Step 5 - Remove store-specific category codes
def clean_item_name(name):
    parts = name.split()
    if parts and SHORT_CAP_PREFIX.match(parts[0]):
        parts = parts[1:]
    return " ".join(parts).strip()


# Step 6a – Grocery extraction
def extract_grocery(rows):
    rows = merge_orphan_name_rows(rows)

    items        = []
    running_cents = 0

    for row in rows:
        price = row["price"]
        name  = (row["name"] or "").strip()

        if price is None:
            continue                 # header / no-price row

        p_cents = cents(price)

        # Running-sum total check
        if p_cents == running_cents and running_cents > 0:
            total = price
            return _build_grocery_result(items, total, "running_sum_match")

        # Discount (negative price)
        if price < 0:
            if items:
                items[-1]["price"] = round(items[-1]["price"] + price, 2)
                items[-1]["adjusted"] = True
                items[-1].setdefault("discounts", []).append(
                    {"label": name or row["row_str"].strip(), "amount": price})
            running_cents += p_cents
            continue

        # Normal item
        if name:
            items.append({"name": clean_item_name(name), "price": price})
        running_cents += p_cents

    # If all rows parsed without finding the total via running sum,
    # the total was not present in the OCR output).
    computed_total = round(running_cents / 100, 2) if running_cents else None
    return _build_grocery_result(items, computed_total, "computed_sum")


def _build_grocery_result(items, total, source):
    return {
        "type":         "grocery",
        "items":        items,
        "total":        total,
        "total_source": source,
    }


# Step 6b – Restaurant extraction
def extract_restaurant(rows):
    rows = merge_orphan_name_rows(rows)

    running_cents = 0
    price_sequence = [(i, r["price"]) for i, r in enumerate(rows)
                      if r["price"] is not None and r["price"] > 0]

    for i, (row_idx, price) in enumerate(price_sequence):
        p_cents = cents(price)
        if p_cents == running_cents and running_cents > 0:
            label = (rows[row_idx]["name"] or rows[row_idx]["row_str"]).strip()
            return {"type": "restaurant", "total": price, "total_label": label}
        running_cents += p_cents

    # Fallback: no running-sum match found.
    # The most-repeated positive price is the total
    if price_sequence:
        freq = defaultdict(int)
        for _, p in price_sequence: freq[cents(p)] += 1
        best_cents = max(freq, key=freq.__getitem__)
        best_price = best_cents / 100
        for row_idx, price in price_sequence:
            if cents(price) == best_cents:
                label = (rows[row_idx]["name"] or rows[row_idx]["row_str"]).strip()
                return {"type": "restaurant", "total": best_price, "total_label": label}

    return {"type": "restaurant", "total": None, "total_label": None}


# Public API
def tokens_from_ocr_dict(data):
    """
    Extract tokens from a raw OCR output dict.

    Parameters
    ----------
    data : dict
        Raw OCR output with keys rec_texts, rec_scores, rec_boxes.

    Returns
    -------
    list of token dicts {text, score, box}
    """
    tokens = []
    for text, score, box in zip(
            data.get("rec_texts", []),
            data.get("rec_scores", []),
            data.get("rec_boxes", [])):
        text = text.strip()
        if text and score >= 0.15:
            tokens.append({"text": text, "score": score, "box": box})
    return tokens


def parse_receipt(sources, receipt_type):
    """
    Full pipeline from OCR outputs to structured result.

    Parameters
    ----------
    sources : list[str | dict]
        - a file path (str) to a pipeline JSON file (str)
        - a raw OCR output dict (dict)

    receipt_type : str  
        "grocery" or "restaurant"

    Returns
    -------
    dict with keys:
        type, items, total, total_source  (grocery)
        type, total, total_label          (restaurant)
    """
    pipeline_row_lists = []
    for src in sources:
        if isinstance(src, dict):
            tokens = tokens_from_ocr_dict(src)
        else:
            tokens = load_pipeline(src)
        pipeline_row_lists.append(parse_pipeline_rows(tokens))

    rows = reconcile_rows(pipeline_row_lists)
    rows = merge_orphan_name_rows(rows)
    if receipt_type == "grocery":
        return extract_grocery(rows)
    elif receipt_type == "restaurant":
        return extract_restaurant(rows)
    else:
        raise ValueError(f"Invalid receipt type.")