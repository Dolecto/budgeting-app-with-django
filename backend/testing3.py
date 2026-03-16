# python-doctr

# import matplotlib.pyplot as plt

from doctr.io import DocumentFile
from doctr.models import ocr_predictor, from_hub
import cv2
from PIL import Image
from ocr_pipeline.ocr_dewarper import Dewarp
from ocr_pipeline.ocr_tiler import get_tiles
import numpy as np

# doc = DocumentFile.from_images("../testing-materials/receipt_deu_03.jpg")
# print(f"Number of pages: {len(doc)}")



# print(predictor)
# result = predictor(doc)
#string_result = result.render()
#print(string_result)


# Instantiate a pretrained model
reco_model = from_hub("Noxilus/doctr-torch-parseq-german")
predictor = ocr_predictor(pretrained=True, assume_straight_pages=False, reco_arch=reco_model)

# TEST WITH TILER
dewarper = Dewarp()

TEST_IMAGE = "../testing-materials/receipt_deu_03.jpg"
image = cv2.imread(TEST_IMAGE)

for tile in get_tiles(image):
    rgb = cv2.cvtColor(tile["image"], cv2.COLOR_BGR2RGB).copy()
    pil_image = Image.fromarray(rgb)
    dewarped = dewarper.inference(pil_image) # still PIL image
    doc = np.array(dewarped)
    #print(doc.shape)
    #print(doc.ndim)

    result = predictor([doc])
    string_result = result.render()

    print(f"TILE NO. {tile["index"]}, TYPE {tile["type"]}")
    print(string_result)
    print("\n-----------------------------------")