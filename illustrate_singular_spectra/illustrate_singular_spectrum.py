import torch
import matplotlib.pyplot as plt
import os
from transformers import ViTForImageClassification


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


def plot_singular_spectrum_and_condition_numbers(model, output_dir="singular_spectrum"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    total_condition_product = 1  # Initialize product of condition numbers
    condition_numbers = []  # List to store condition numbers for each layer

    # Step 1: Read all weight matrices (or tensors) from the model
    weight_matrices = []
    for name, param in model.named_parameters():
        if (
            "weight" in name and param.dim() == 2
        ):  # Only consider 2D weight matrices (ignore tensors with higher order)
            weight_matrices.append((name, param.detach().cpu().numpy()))
        else:
            print(f"Skipping {name} (not a 2D matrix)")

    # Step 2 & 3: Compute SVD, plot singular values, and compute condition numbers
    for idx, (param_name, weight_matrix) in enumerate(weight_matrices):
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

        # Compute condition number as ratio of largest to smallest singular value
        condition_number = s.max() / s.min()
        condition_numbers.append(condition_number)  # Store the condition number
        print(f"Condition number of {param_name} (Layer {idx}): {condition_number}")

        # Multiply to the total condition product
        total_condition_product *= condition_number

    # Step 4: Plot condition numbers over layer index
    plt.figure()
    plt.plot(
        range(len(condition_numbers)), condition_numbers, marker="o", linestyle="-"
    )
    plt.title("Condition Numbers Over Layer Index")
    plt.xlabel("Layer Index")
    plt.ylabel("Condition Number")
    plt.yscale("log")  # Log scale for better visualization of large condition numbers
    plt.grid(True)

    # Save the condition number plot
    plt.savefig(os.path.join(output_dir, "condition_numbers_over_layers.png"))
    plt.close()

    # Print the product of all condition numbers
    print(f"Product of all condition numbers: {total_condition_product}")


# Load a pretrained ViT model from Hugging Face
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True,
)
# print(model)
# exit(1)
# Perform singular spectrum analysis on the model
# plot_singular_spectrum(model)
plot_singular_spectrum_and_condition_numbers(model)
