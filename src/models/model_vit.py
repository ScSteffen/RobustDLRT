# %%
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import ViTForImageClassification
import torch.nn as nn
from transformers import ViTModel


def vit_b16(n_classes=10):
    """
    Load a pre-trained Vision Transformer (ViT) model for image classification
    and configure it for a specific number of classes.

    Parameters:
    - n_classes (int): Number of classes for the classification task. Default is 10.

    Returns:
    - model (torch.nn.Module): Configured ViT model for image classification.
    """
    # === pytorch version =====
    # Load the pre-trained ViT model with the specified configuration
    # weights = ViT_B_16_Weights.DEFAULT  # Use the default pre-trained weights
    # model = vit_b_16(weights=weights)
    #
    ## Modify the classifier head to match the number of output classes
    # model.heads.head = torch.nn.Linear(model.heads.head.in_features, n_classes)

    # ==== transformers version
    model = ViTForImageClassification.from_pretrained(
        # google/vit-base-patch32-224-in21k
        # google/vit-huge-patch14-224-in21k
        "google/vit-base-patch16-224",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def vit_l32(n_classes=10):
    """
    Load a pre-trained Vision Transformer (ViT) model for image classification
    and configure it for a specific number of classes.

    Parameters:
    - n_classes (int): Number of classes for the classification task. Default is 10.

    Returns:
    - model (torch.nn.Module): Configured ViT model for image classification.
    """
    # === pytorch version =====
    # Load the pre-trained ViT model with the specified configuration
    # weights = ViT_B_16_Weights.DEFAULT  # Use the default pre-trained weights
    # model = vit_b_16(weights=weights)
    #
    ## Modify the classifier head to match the number of output classes
    # model.heads.head = torch.nn.Linear(model.heads.head.in_features, n_classes)

    # ==== transformers version
    model = ViTForImageClassification.from_pretrained(
        # google/vit-base-patch32-224-in21k
        # google/vit-base-patch16-224
        # google/vit-huge-patch14-224-in21k
        "google/vit-large-patch32-224-in21k",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def vit_l16(n_classes=10):
    """
    Load a pre-trained Vision Transformer (ViT) model for image classification
    and configure it for a specific number of classes.

    Parameters:
    - n_classes (int): Number of classes for the classification task. Default is 10.

    Returns:
    - model (torch.nn.Module): Configured ViT model for image classification.
    """
    # === pytorch version =====
    # Load the pre-trained ViT model with the specified configuration
    # weights = ViT_B_16_Weights.DEFAULT  # Use the default pre-trained weights
    # model = vit_b_16(weights=weights)
    #
    ## Modify the classifier head to match the number of output classes
    # model.heads.head = torch.nn.Linear(model.heads.head.in_features, n_classes)

    # ==== transformers version
    model = ViTForImageClassification.from_pretrained(
        # google/vit-base-patch32-224-in21k
        # google/vit-base-patch16-224
        # google/vit-huge-patch14-224-in21k
        "google/vit-large-patch16-224",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def vit_h14(n_classes=10):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-huge-patch14-224-in21k",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def vit_g14(n_classes=10):
    model = ViTForImageClassification.from_pretrained(
        "facebook/dinov2-giant",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    model.config.image_size = 224
    return model


class ViTSegmentationModel(nn.Module):
    def __init__(self, num_classes=7, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()
        # Load a pre-trained Vision Transformer as the encoder (without classification head)
        self.encoder = ViTModel.from_pretrained(pretrained_name)
        self.num_classes = num_classes

        # Decoder: transforms ViT features to full-resolution segmentation map
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample from 14x14 → 56x56 (ViT base has 14x14 tokens for 224x224 input)
            nn.Upsample(
                scale_factor=4, mode="bilinear", align_corners=False
            ),  # 14x14 → 56x56
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Upsample from 56x56 → 224x224 (original input resolution)
            nn.Upsample(
                scale_factor=4, mode="bilinear", align_corners=False
            ),  # 56x56 → 224x224
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        # Extract patch embeddings and reshape to spatial grid
        # Usually outputs a sequence, but I think we need a spatial feature map
        outputs = self.encoder(x)
        print(outputs.last_hidden_state.shape)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # remove CLS token

        B, N, C = patch_tokens.shape
        H = W = int(N**0.5)
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)  # (B, 768, H, W)
        x = self.decoder(x)  # (B, num_classes, H*4, W*4)
        return x


def vit_b16_seg(num_classes=7):
    return ViTSegmentationModel(num_classes=num_classes)


class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return self.upsample(out)
    
class ViTSegmentationRVSA(nn.Module):
    def __init__(self, num_classes=7, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(pretrained_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.decoder1 = ResidualDecoderBlock(768, 256, upsample_factor=2)   # 14→28
        self.decoder2 = ResidualDecoderBlock(256, 128, upsample_factor=2)   # 28→56
        self.decoder3 = ResidualDecoderBlock(128, 64, upsample_factor=2)    # 56→112
        self.decoder4 = ResidualDecoderBlock(64, 32, upsample_factor=2)     # 112→224
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.size(0)
        outputs = self.encoder(x)
        patch_tokens = outputs.last_hidden_state[:, 1:]  # remove CLS
        x = patch_tokens.permute(0, 2, 1).reshape(B, self.hidden_size, 14, 14)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        return self.final_conv(x)
    
def vit_b16_rvsa(num_classes=7):
    return ViTSegmentationRVSA(num_classes=num_classes)

def oreole_b16_rvsa(num_classes=7):
    return ViTSegmentationRVSA(num_classes=num_classes, pretrained_name="hf_vitB_from_oreole")
