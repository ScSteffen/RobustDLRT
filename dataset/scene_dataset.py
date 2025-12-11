from torch.utils.data import Dataset
from PIL import Image

import os


def default_loader(path):
    return Image.open(path).convert("RGB")


class scene_dataset(Dataset):

    def __init__(
        self,
        root_dir,
        pathfile,
        transform=None,
        loader=default_loader,
        mode="clean",
        classname=None,
    ):
        # Jerry rigged solution, classname is messing up file selection, fix it later for better compatibility
        classname = None
        pf = open(pathfile, "r")
        imgs = []
        if mode == "clean":
            for line in pf:
                line = line.rstrip("\n")
                words = line.split()
                if classname:
                    name = (
                        classname[int(words[1])]
                        + "_"
                        + words[0].split("/")[-1].split(".")[0]
                    )
                else:
                    name = words[0].split("/")[-1].split(".")[0]

                imgs.append((root_dir + words[0], int(words[1]), name))
        elif mode == "adv":
            for line in pf:
                line = line.rstrip("\n")
                words = line.split()

                path = root_dir + words[0].split("/")[-1].split(".")[0] + "_adv.png"
                if classname:
                    name = classname[int(words[1])]
                    dir_path, filename = os.path.split(path)

                    # Remove any existing class name prefix from the filename
                    # e.g., from 'agricultural28_adv.png', remove 'agricultural' if present
                    for cls in classname:
                        if filename.startswith(cls + "_"):
                            filename = filename[
                                len(cls) + 1 :
                            ]  # strip 'classname_' prefix

                    new_filename = f"{name}_{filename}"
                    path = os.path.join(dir_path, new_filename)

                else:
                    name = words[0].split("/")[-1].split(".")[0]

                imgs.append(
                    (
                        path,
                        int(words[1]),
                        name,
                    )
                )

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        pf.close()

    def __getitem__(self, index):
        fn, label, name = self.imgs[index]
        img = self.loader(fn)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, name

    def __len__(self):
        return len(self.imgs)
