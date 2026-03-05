import torch
from transformers import (
    Sam3Processor,
    Sam3Model,
    Sam3TrackerProcessor,
    Sam3TrackerModel,
)


class sam3_predictor:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.processor = Sam3Processor.from_pretrained(model_dir)
        self.model = Sam3Model.from_pretrained(model_dir).to(device)

    def pred_on_prompts_and_single_img(
        self, img, prompts, threshold=0.5, mask_threshold=0.5
    ):
        inputs = self.processor(
            images=[img] * len(prompts), text=prompts, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )

        return results
