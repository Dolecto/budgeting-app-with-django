import os
from functools import partial
from ocr_pipeline.ocr_preprocessing import OCRPreprocessingPipeline as Pipeline
from ocr_pipeline.ocr_text_extraction import load_ocr, extract_text


# TODO: Connect to API.
input_path = "../testing-materials/receipt_eng_04.jpg"
img_basename = os.path.splitext(os.path.basename(input_path))[0]


pipeline_1 = Pipeline(
    steps=[
        partial(Pipeline.upscale, scale=1.15),
        partial(Pipeline.denoise, method="bilateral"),
        partial(Pipeline.enhance, method="clahe"),
        partial(Pipeline.denoise, method="bilateral"),
        partial(Pipeline.sharpen, method="unsharp", strength=0.5),
    ], name="pipeline_1")


pipeline_2 = Pipeline(
    steps=[
        #Pipeline.normalize,
            partial(Pipeline.denoise, method="bilateral"),
            partial(Pipeline.enhance, method="clahe"),
            partial(Pipeline.upscale, scale=1.15)
    ], name="pipeline_2")


pipeline_3 = Pipeline(
    steps=[
        #Pipeline.normalize,
        partial(Pipeline.denoise, method="gaussian"),
        partial(Pipeline.enhance, method="clahe"),
        # partial(Pipeline.binarize, method="otsu"),
    ], name="pipeline_3")


pipeline_4 = Pipeline(
    steps=[
        #Pipeline.normalize,
        partial(Pipeline.denoise, method="nlm"),
        partial(Pipeline.enhance, method="clahe"),
        partial(Pipeline.upscale, scale=1.15),
    ], name="pipeline_4")


pipeline_5 = Pipeline(
    steps=[
        partial(Pipeline.upscale, scale=1.15),
        partial(Pipeline.denoise, method="nlm"),
        partial(Pipeline.enhance, method="clahe"),
        # partial(Pipeline.denoise, method="bilateral"),
        partial(Pipeline.sharpen, method="unsharp", strength=0.5),
    ], name="pipeline_5")


ocr_model = load_ocr()


for pipeline in [
    pipeline_1, 
    pipeline_2, 
    pipeline_3, 
    pipeline_4, 
    pipeline_5, 
    ]:
    extract_text(
        ocr_model=ocr_model,
        img=pipeline.run(input_path), 
        json_output_dir="json_outputs", 
        filename=f"{img_basename}_{pipeline.name}",
        debug=True, 
        debug_dir="ocr_debugging"
    )