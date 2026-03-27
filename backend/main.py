from functools import partial
from ocr_pipeline.ocr_preprocessing import OCRPreprocessingPipeline as Pipeline
from ocr_pipeline.ocr_text_extraction import extract_text as ocr


# TODO: Connect to API.
input_path = "../testing-materials/receipt_deu_04.jpg"


pipeline_1 = Pipeline(steps=[
    partial(Pipeline.upscale, scale=1.15),
    partial(Pipeline.denoise, method="bilateral"),
    partial(Pipeline.enhance, method="clahe+bilateral+unsharp"),
])


pipeline_2 = Pipeline(steps=[
    Pipeline.normalize,
    partial(Pipeline.denoise, method="bilateral"),
    partial(Pipeline.enhance, method="clahe"),
    partial(Pipeline.upscale, scale=1.15)
])


result_1 = ocr(pipeline_1.run(input_path), 
               json_output_dir="json_outputs", 
               debug=True, 
               debug_dir="ocr_debugging"
               )

result_2 = ocr(pipeline_2.run(input_path), 
               json_output_dir="json_outputs", 
               debug=True, 
               debug_dir="ocr_debugging"
               )