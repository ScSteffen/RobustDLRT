import os
from glob import glob


def get_imagenet_classnames(root_dir):
    train_dir = os.path.join(root_dir, "train")
    return sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())


#
