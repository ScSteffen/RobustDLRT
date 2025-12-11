import os
from glob import glob


def create_imagenet_split_txt(
    root_dir, split="train", out_file="dataset/imagenet_train.txt"
):
    split_dir = os.path.join(root_dir, split)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Sorted list of class folders
    class_folders = sorted(
        entry.name for entry in os.scandir(split_dir) if entry.is_dir()
    )
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

    with open(out_file, "w") as f:
        for cls_name in class_folders:
            class_dir = os.path.join(split_dir, cls_name)
            img_paths = glob(os.path.join(class_dir, "*"))
            for img_path in img_paths:
                class_idx = class_to_idx[cls_name]
                f.write(f"{img_path} {class_idx}\n")

    print(
        f"Wrote {out_file} with {len(class_folders)} classes and {sum(len(glob(os.path.join(os.path.join(split_dir, cls), '*'))) for cls in class_folders)} images."
    )


# Example usage:
create_imagenet_split_txt(
    root_dir="./imagenet",
    split="train",
    out_file="../imagenet_train.txt",
)  #
create_imagenet_split_txt(
    root_dir="./imagenet",
    split="val",
    out_file="../imagenet_val.txt",
)
