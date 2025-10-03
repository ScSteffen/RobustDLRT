import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import ViTForImageClassification
import torch.nn as nn
import math


# from transformers import ViTModel
from src.tools.high_res_vit import load_weights_from_hf


class ViTSegmentationModelHighRes(nn.Module):
    def __init__(
        self, num_classes=7, patchsize=16, pretrained_name="google/vit-base-patch16-224"
    ):
        super().__init__()
        # Load a pre-trained Vision Transformer as the encoder (without classification head)
        self.encoder, _ = load_weights_from_hf(patchsize=patchsize)
        self.num_classes = num_classes

        self.patch_size = patchsize

        self.encoder_output_size = 224 // self.patch_size
        if 224 % self.patch_size != 0:
            raise RuntimeError(
                f"Image size 224 must be divisible by patch size {self.patch_size}"
            )
        if self.patch_size == 16:
            # Decoder: transforms ViT features to full-resolution segmentation map
            self.decoder = nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 14x14 → 56x56 (ViT base has 14x14 tokens for 224x224 input)
                nn.Upsample(
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                ),  # 14x14 → 56x56
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 56x56 → 224x224 (original input resolution)
                nn.Upsample(
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                ),  # 56x56 → 224x224
                nn.Conv2d(128, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        elif self.patch_size == 8:
            # Decoder: transforms ViT features to full-resolution segmentation map
            self.decoder = nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 14x14 → 56x56 (ViT base has 14x14 tokens for 224x224 input)
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                ),  # 14x14 → 56x56
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 56x56 → 224x224 (original input resolution)
                nn.Upsample(
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                ),  # 56x56 → 224x224
                nn.Conv2d(128, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        elif self.patch_size == 4:
            # Decoder: transforms ViT features to full-resolution segmentation map
            self.decoder = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 14x14 → 56x56 (ViT base has 14x14 tokens for 224x224 input)
                nn.Upsample(
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                ),  # 14x14 → 224x224
                nn.Conv2d(128, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        elif self.patch_size == 2:
            self.decoder = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 14x14 → 56x56 (ViT base has 14x14 tokens for 224x224 input)
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                ),  # 14x14 → 224x224
                nn.Conv2d(128, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        elif self.patch_size == 1:
            self.decoder = nn.Sequential(
                nn.Conv2d(768, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # Upsample from 14x14 → 56x56 (ViT base has 14x14 tokens for 224x224 input)
                nn.Conv2d(128, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        # x: (B, 3, H, W)
        # Extract patch embeddings and reshape to spatial grid
        # Usually outputs a sequence, but I think we need a spatial feature map
        outputs = self.encoder(x)

        patch_tokens = outputs[:, 1:]

        B, N, C = patch_tokens.shape
        H = W = int(N**0.5)
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 768, H, W)
        # print(x.shape)
        x = self.decoder(x)  # (B, num_classes, H*4, W*4)
        return x


def vit_seg_high_res(num_classes=7, patchsize=16):
    return ViTSegmentationModelHighRes(num_classes=num_classes, patchsize=patchsize)
