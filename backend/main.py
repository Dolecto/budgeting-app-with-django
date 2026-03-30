import os
from functools import partial
from ocr_pipeline.ocr_preprocessing import OCRPreprocessingPipeline as Pipeline
from ocr_pipeline.ocr_text_extraction import load_ocr, extract_text


# TODO: Connect to API.
input_path = "../testing-materials/receipt_eng_03.jpg"
img_basename = os.path.splitext(os.path.basename(input_path))[0]


pipeline_1 = Pipeline(steps=[
    partial(Pipeline.upscale, scale=1.15),
    partial(Pipeline.denoise, method="bilateral"),
    partial(Pipeline.enhance, method="clahe+bilateral+unsharp"),
], name="pre-proc-1")


pipeline_2 = Pipeline(steps=[
    Pipeline.normalize,
    partial(Pipeline.denoise, method="bilateral"),
    partial(Pipeline.enhance, method="clahe"),
    partial(Pipeline.upscale, scale=1.15)
], name="pre-proc-2")

ocr_model = load_ocr()

for pipeline in [pipeline_1, pipeline_2]:
    extract_text(
        ocr_model=ocr_model,
        img=pipeline.run(input_path), 
        json_output_dir="json_outputs", 
        filename=f"{img_basename}_{pipeline.name}",
        debug=True, 
        debug_dir="ocr_debugging"
    )