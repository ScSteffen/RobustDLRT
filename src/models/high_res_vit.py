import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformers
import collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def load_weights_from_hf(patchsize=16):
    # Load pretrained HF model
    hf_model = transformers.ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )

    # Create our model and load weights
    custom_model = ViTModel(
        img_size=224,
        patch_size=patchsize,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        add_pooling_layer=False,
        use_mask_token=False,
        dropout_prob=0.0,
    )

    # print(hf_model)
    # print(custom_model)

    # custom_model.load_state_dict(hf_model.vit.state_dict(), strict=False)
    # Filter out mismatched keys
    hf_state_dict = hf_model.vit.state_dict()
    model_state_dict = custom_model.state_dict()

    ingored_keys = [
        "embeddings.position_embeddings",
        "embeddings.patch_embeddings.projection.weight",
    ]
    filtered_state_dict = {}
    for k, v in hf_state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            if not any(substr in k for substr in ingored_keys):
                filtered_state_dict[k] = v

    # Load only matching weights
    load_info = custom_model.load_state_dict(filtered_state_dict, strict=False)

    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)
    return custom_model, hf_model


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        image_size, patch_size = img_size, patch_size
        num_channels, hidden_size = in_chans, embed_dim

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTEmbeddings(nn.Module):

    def __init__(
        self, img_size, patch_size, in_chans, embed_dim, dropout_prob, use_mask_token
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_mask_token else None
        )
        self.patch_embeddings = ViTPatchEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        embeddings = self.patch_embeddings(x, interpolate_pos_encoding=None)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # print(embeddings.shape)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # print(embeddings.shape)
        embeddings = embeddings + self.position_embeddings
        # print(embeddings.shape)
        return self.dropout(embeddings)


class ViTSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_prob):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(embed_dim, self.all_head_size, bias=True)
        self.key = nn.Linear(embed_dim, self.all_head_size, bias=True)
        self.value = nn.Linear(embed_dim, self.all_head_size, bias=True)

    def forward(self, x):
        key_layer = self.transpose_for_scores(self.key(x))
        value_layer = self.transpose_for_scores(self.value(x))
        query_layer = self.transpose_for_scores(self.query(x))

        context_layer, _ = self.eager_attention_forward(
            query_layer,
            key_layer,
            value_layer,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

        # Normalize the attention scores to probabilities.
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=self.training
        )

        # Mask heads if we want to
        if attention_mask is not None:
            exit("careful, attention mask used")
            attn_weights = attn_weights * attention_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, embed_dim, dropout_prob) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.dropout(x)

        return x


class ViTAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_prob) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout_prob=dropout_prob
        )
        self.output = ViTSelfOutput(embed_dim=embed_dim, dropout_prob=dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.attention(x)
        x = self.output(x)
        return x


class ViTIntermediate(nn.Module):

    def __init__(self, embed_dim, mlp_ratio) -> None:
        super().__init__()
        self.dense = nn.Linear(embed_dim, mlp_ratio * embed_dim)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.intermediate_act_fn(x)

        return x


class ViTOutput(nn.Module):
    def __init__(self, embed_dim, mlp_ratio) -> None:
        super().__init__()
        self.dense = nn.Linear(mlp_ratio * embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        return x


class ViTLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout_prob):
        super().__init__()

        self.seq_len_dim = 1
        self.attention = ViTAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout_prob=dropout_prob
        )
        self.intermediate = ViTIntermediate(embed_dim=embed_dim, mlp_ratio=mlp_ratio)
        self.output = ViTOutput(embed_dim=embed_dim, mlp_ratio=mlp_ratio)

        self.layernorm_before = nn.LayerNorm(embed_dim, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(embed_dim, eps=1e-12)

    def forward(self, x):
        x = x + self.attention(self.layernorm_before(x))
        x = x + self.output(self.intermediate(self.layernorm_after(x)))
        return x


class ViTEncoder(nn.Module):

    def __init__(self, depth, embed_dim, num_heads, mlp_ratio, dropout_prob):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                ViTLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_prob=dropout_prob,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class ViTModel(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio,
        add_pooling_layer,
        use_mask_token,
        dropout_prob,
    ):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embeddings = ViTEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_mask_token=use_mask_token,
            dropout_prob=dropout_prob,
        )
        self.encoder = ViTEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )

        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)

    def forward(self, x):

        x = self.embeddings(x)

        x = self.encoder(x)

        x = self.layernorm(x)
        # print(x)
        # exit("vit")
        return x


class ViTForImageClassification(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
    ):
        super().__init__()

        self.num_labels = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.vit = ViTModel(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            add_pooling_layer=False,
            use_mask_token=False,
            dropout_prob=0.0,
        )

        # Classifier head
        self.classifier = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x):
        x = self.vit(x)

        return self.classifier(x)  # use CLS token
