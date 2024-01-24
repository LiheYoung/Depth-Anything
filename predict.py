# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from cog import BasePredictor, Input, Path

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        encoder_options = ["vits", "vitb", "vitl"]
        self.device = "cuda:0"
        model_cache = "model_cache"
        self.models = {
            k: DepthAnything.from_pretrained(
                f"LiheYoung/depth_anything_{k}14", cache_dir=model_cache
            ).to(self.device)
            for k in encoder_options
        }
        self.total_params = {
            k: sum(param.numel() for param in self.models[k].parameters())
            for k in encoder_options
        }

        self.transform = Compose(
            [
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        encoder: str = Input(
            description="Choose an encoder.",
            default="vitl",
            choices=["vits", "vitb", "vitl"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        depth_anything = self.models[encoder]
        total_params = self.total_params[encoder]
        print("Total parameters: {:.2f}M".format(total_params / 1e6))

        depth_anything.eval()

        raw_image = cv2.imread(str(image))
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = self.transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        output_path = "/tmp/out.png"
        cv2.imwrite(output_path, depth_color)

        return Path(output_path)
