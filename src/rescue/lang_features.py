from pathlib import Path
from typing import Union, Optional

import clip
import cv2
import torch

from lseg import LSegNet


class LSegLangFeatures:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        backbone: str = "clip_vitl16_384",
        num_features: int = 256,
        arch_option: int = 0,
        block_depth: int = 0,
        activation: str = "lrelu",
        crop_size: int = 480,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = LSegNet(
            backbone=backbone,
            features=num_features,
            crop_size=crop_size,
            arch_option=arch_option,
            block_depth=block_depth,
            activation=activation,
        )
        self.net.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
        self.net.eval()
        self.net.to(self.device)

        self.clip_text_encoder = self.net.clip_pretrained.encode_text
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def extract_dense_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_feat = self.net.forward(image_tensor)
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
        return img_feat_norm


    def extract_dense_features(self, image_path: Union[str, Path]) -> torch.Tensor:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel if present
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)  # 1, C, H, W

        with torch.no_grad():
            img_feat = self.net.forward(img)
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
        return img_feat_norm

    def match_text(
        self, image_path: Union[str, Path], prompt: str
    ) -> torch.Tensor:
        img_feat_norm = self.extract_dense_features(image_path)
        tokenized_prompt = clip.tokenize(prompt).to(self.device)
        with torch.no_grad():
            text_feat = self.clip_text_encoder(tokenized_prompt)
            text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)
            similarity = self.cosine_similarity(
                img_feat_norm, text_feat_norm.unsqueeze(-1).unsqueeze(-1)
            )
        return similarity

    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        tokenized_prompt = clip.tokenize(prompt).to(self.device)
        with torch.no_grad():
            text_feat = self.clip_text_encoder(tokenized_prompt)
            text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)
        return text_feat_norm.float()