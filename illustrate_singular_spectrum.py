import torch
import matplotlib.pyplot as plt
import os
import timm


def plot_singular_spectrum(model, output_dir="singular_spectrum"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read all weight matrices (or tensors) from the model
    weight_matrices = []
    for name, param in model.named_parameters():
        if (
            "weight" in name and param.dim() >= 2
        ):  # Only consider weight matrices (not biases)
            weight_matrices.append((name, param.detach().cpu().numpy()))

    # Step 2 & 3: Compute SVD and plot singular values
    for param_name, weight_matrix in weight_matrices:
        # Compute the SVD
        u, s, vh = torch.linalg.svd(torch.tensor(weight_matrix), full_matrices=False)

        # Plot the singular values
        plt.figure()
        plt.plot(range(1, len(s) + 1), s, marker="o", linestyle="-")
        plt.title(f"Singular Spectrum of {param_name}")
        plt.xlabel("Index of Singular Value")
        plt.ylabel("Singular Value")
        plt.yscale("log")  # Log scale to capture wide ranges of values
        plt.grid(True)

        # Save the plot
        output_path = os.path.join(output_dir, f"{param_name}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Singular spectrum plots saved in: {output_dir}")


# Example usage:
# model = torch.load('your_pretrained_model.pth')  # Load your model
# plot_singular_spectrum(model)

# Load the pretrained Vision Transformer (ViT) model from timm
model = timm.create_model("vit_base_patch32_224", pretrained=True)

# Perform singular spectrum analysis on the model
plot_singular_spectrum(model)
