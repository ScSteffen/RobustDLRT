import argparse
import torch
from transformers import ViTForImageClassification, ViTConfig
import os


def split_qkv_weights(timm_state_dict, hidden_size, num_layers, num_labels):
    hf_state_dict = {}

    for i in range(num_layers):
        prefix = f"layers.{i}"

        qkv_w = timm_state_dict.get(f"{prefix}.attn.attn.in_proj_weight")
        qkv_b = timm_state_dict.get(f"{prefix}.attn.attn.in_proj_bias")
        if qkv_w is not None:
            q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
            hf_state_dict[f"vit.encoder.layer.{i}.attention.attention.query.weight"] = q_w
            hf_state_dict[f"vit.encoder.layer.{i}.attention.attention.key.weight"] = k_w
            hf_state_dict[f"vit.encoder.layer.{i}.attention.attention.value.weight"] = v_w
        if qkv_b is not None:
            q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
            hf_state_dict[f"vit.encoder.layer.{i}.attention.attention.query.bias"] = q_b
            hf_state_dict[f"vit.encoder.layer.{i}.attention.attention.key.bias"] = k_b
            hf_state_dict[f"vit.encoder.layer.{i}.attention.attention.value.bias"] = v_b

        hf_state_dict[f"vit.encoder.layer.{i}.attention.output.dense.weight"] = timm_state_dict[f"{prefix}.attn.attn.out_proj.weight"]
        hf_state_dict[f"vit.encoder.layer.{i}.attention.output.dense.bias"] = timm_state_dict[f"{prefix}.attn.attn.out_proj.bias"]

        hf_state_dict[f"vit.encoder.layer.{i}.layernorm_before.weight"] = timm_state_dict[f"{prefix}.ln1.weight"]
        hf_state_dict[f"vit.encoder.layer.{i}.layernorm_before.bias"] = timm_state_dict[f"{prefix}.ln1.bias"]

        hf_state_dict[f"vit.encoder.layer.{i}.intermediate.dense.weight"] = timm_state_dict[f"{prefix}.ffn.layers.0.0.weight"]
        hf_state_dict[f"vit.encoder.layer.{i}.intermediate.dense.bias"] = timm_state_dict[f"{prefix}.ffn.layers.0.0.bias"]

        hf_state_dict[f"vit.encoder.layer.{i}.output.dense.weight"] = timm_state_dict[f"{prefix}.ffn.layers.1.weight"]
        hf_state_dict[f"vit.encoder.layer.{i}.output.dense.bias"] = timm_state_dict[f"{prefix}.ffn.layers.1.bias"]

        hf_state_dict[f"vit.encoder.layer.{i}.layernorm_after.weight"] = timm_state_dict[f"{prefix}.ln2.weight"]
        hf_state_dict[f"vit.encoder.layer.{i}.layernorm_after.bias"] = timm_state_dict[f"{prefix}.ln2.bias"]

    hf_state_dict["vit.embeddings.patch_embeddings.projection.weight"] = timm_state_dict["patch_embed.projection.weight"]
    hf_state_dict["vit.embeddings.patch_embeddings.projection.bias"] = timm_state_dict["patch_embed.projection.bias"]
    hf_state_dict["vit.embeddings.position_embeddings"] = timm_state_dict["pos_embed"]
    hf_state_dict["vit.embeddings.cls_token"] = timm_state_dict["cls_token"]

    if "ln_post.weight" in timm_state_dict:
        hf_state_dict["vit.layernorm.weight"] = timm_state_dict["ln_post.weight"]
        hf_state_dict["vit.layernorm.bias"] = timm_state_dict["ln_post.bias"]
    elif "norm.weight" in timm_state_dict:
        hf_state_dict["vit.layernorm.weight"] = timm_state_dict["norm.weight"]
        hf_state_dict["vit.layernorm.bias"] = timm_state_dict["norm.bias"]
    else:
        print("Final layernorm not found in checkpoint — initializing manually.")
        hf_state_dict["vit.layernorm.weight"] = torch.ones(hidden_size)
        hf_state_dict["vit.layernorm.bias"] = torch.zeros(hidden_size)

    if "head.weight" in timm_state_dict:
        hf_state_dict["classifier.weight"] = timm_state_dict["head.weight"]
        hf_state_dict["classifier.bias"] = timm_state_dict["head.bias"]
    else:
        print("No classifier head found in checkpoint — model will initialize it randomly.")

    return hf_state_dict


def get_vit_config(model_size: str, hidden_size: int, num_layers: int, num_labels: int) -> ViTConfig:
    if model_size == "base":
        return ViTConfig.from_pretrained(
            "google/vit-base-patch16-224",
            torch_dtype=torch.float32,
            num_labels=num_labels,
        )
    elif model_size == "huge":
        return ViTConfig.from_pretrained(
            "google/vit-huge-patch14-224-in21k",
            torch_dtype=torch.float32,
            num_labels=num_labels,
        )
    elif model_size == "giant":
        return ViTConfig(
            hidden_size=1536,
            num_hidden_layers=32,
            num_attention_heads=16,
            intermediate_size=6144,
            patch_size=14,
            image_size=224,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
            classifier="token",
            num_labels=num_labels,
        )
    else:
        raise ValueError(f"Invalid model_size '{model_size}'. Choose from 'base', 'huge', 'giant'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to TIMM checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HF model")
    parser.add_argument("--num_labels", type=int, required=True, help="Number of output classes")
    parser.add_argument("--model_size", type=str, choices=["base", "huge", "giant"], required=True,
                        help="Model size: base, huge, or giant")

    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")

    layer_keys = [k for k in state_dict if k.startswith("layers.") and ".ln1.weight" in k]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    print(f"Detected {num_layers} transformer layers.")

    qkv_weight = state_dict["layers.0.attn.attn.in_proj_weight"]
    hidden_size = qkv_weight.shape[1]
    print(f"Inferred hidden size: {hidden_size}")

    hf_state_dict = split_qkv_weights(state_dict, hidden_size, num_layers, args.num_labels)

    config = get_vit_config(args.model_size, hidden_size, num_layers, args.num_labels)

    model = ViTForImageClassification(config)
    print("Loading converted weights...")
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    os.makedirs(args.output_dir, exist_ok=True)
    model.config.save_pretrained(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"Saved HuggingFace model to: {args.output_dir}")


if __name__ == "__main__":
    main()
