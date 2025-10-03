import torchvision.transforms as T
import torchvision.datasets as datasets
import torch
import os
from src.utils import imagenet_utils
from torch import nn
from torch.utils import data
from torchvision import transforms
from dataset.scene_dataset import scene_dataset


def get_dataset_info(args):

    if args.dataID == 1:
        DataName = "UCM"
        num_classes = 21
        classname = (
            "agricultural",
            "airplane",
            "baseballdiamond",
            "beach",
            "buildings",
            "chaparral",
            "denseresidential",
            "forest",
            "freeway",
            "golfcourse",
            "harbor",
            "intersection",
            "mediumresidential",
            "mobilehomepark",
            "overpass",
            "parkinglot",
            "river",
            "runway",
            "sparseresidential",
            "storagetanks",
            "tenniscourt",
        )
    elif args.dataID == 2:
        DataName = "AID"
        args.root_dir = "./dataset/data_adversarial_rs/AID/"
        num_classes = 30
        classname = (
            "Airport",
            "BareLand",
            "BaseballField",
            "Beach",
            "Bridge",
            "Center",
            "Church",
            "Commercial",
            "DenseResidential",
            "Desert",
            "Farmland",
            "Forest",
            "Industrial",
            "Meadow",
            "MediumResidential",
            "Mountain",
            "Parking",
            "Park",
            "Playground",
            "Pond",
            "Port",
            "RailwayStation",
            "Resort",
            "River",
            "School",
            "SparseResidential",
            "Square",
            "Stadium",
            "StorageTanks",
            "Viaduct",
        )
    elif args.dataID == 3:
        DataName = "Cifar10"
        num_classes = 10
        classname = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    elif args.dataID == 4:
        DataName = "MillionAID"
        classname = sorted(
            os.listdir(
                os.path.join(os.path.join(args.root_dir, "/MillionAID/images"), "train")
            )
        )
        # for i in range(len(classname)):
        # print(classname[i])
        num_classes = len(classname)
    elif args.dataID == 5:
        DataName = "NWPU"
        classname = sorted(
            os.listdir(
                os.path.join(os.path.join(args.root_dir, "/NWPU/images"), "train")
            )
        )
        num_classes = len(classname)
    elif args.dataID == 6:
        DataName = "ImageNet"

        num_classes = 1000
        # ImageNet class names are typically loaded from a file or a library, but for simplicity, we use a placeholder tuple
        classname = imagenet_utils.get_imagenet_classnames(args.root_dir)
    else:
        raise NotImplementedError
    return classname, num_classes, DataName


def get_dataloader(args, DataName, classname):
    if args.dataID in [1, 2, 4, 5]:
        composed_transforms = transforms.Compose(
            [
                transforms.Resize(size=(args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        train_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_train.txt",
                transform=composed_transforms,
                # classname=classname,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("Pathfile: ./dataset/" + DataName + "_test.txt")
        val_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
                # classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataID == 3:  # Cifar10
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        import torchvision.transforms as T

        train_transform = T.Compose(
            [
                T.Resize(size=(args.crop_size, args.crop_size)),
                T.RandomCrop(
                    size=args.crop_size,
                    padding=4,
                    padding_mode="reflect",
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*stats, inplace=True),
            ]
        )
        valid_transform = T.Compose(
            [
                T.Resize(size=(args.crop_size, args.crop_size)),
                T.ToTensor(),
                T.Normalize(*stats),
            ]
        )

        train_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_train.txt",
                transform=train_transform,
                # classname=classname,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        val_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=valid_transform,
                # classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataID == 6:
        print("[INFO] Loading ImageNet dataset")
        composed_transforms = transforms.Compose(
            [
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        train_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_train.txt",
                transform=composed_transforms,
                classname=classname,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=True,
        )

        val_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_val.txt",
                transform=composed_transforms,
                classname=classname,
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=True,
        )
    else:
        raise NotImplementedError

    return train_loader, val_loader


def get_adv_valloader(args, DataName, classname, adv_root_dir):
    if args.dataID in [1, 2, 4, 5]:
        from torchvision import transforms

        composed_transforms = transforms.Compose(
            [
                transforms.Resize(size=(args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        adv_loader = data.DataLoader(
            scene_dataset(
                root_dir=adv_root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
                classname=classname,
                mode="adv",
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        clean_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
                classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataID == 3:
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        import torchvision.transforms as T

        valid_transform = T.Compose(
            [
                transforms.Resize(size=(args.crop_size, args.crop_size)),
                T.ToTensor(),
                T.Normalize(*stats),
            ]
        )
        adv_loader = data.DataLoader(
            scene_dataset(
                root_dir=adv_root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=valid_transform,
                mode="adv",
                classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        clean_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.root_dir,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=valid_transform,
                classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    elif args.dataID == 6:
        from torchvision import transforms

        print("[INFO] Loading ImageNet dataset")
        valid_transform = transforms.Compose(
            [
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        adv_loader = data.DataLoader(
            scene_dataset(
                root_dir=adv_root_dir,
                pathfile="./dataset/" + DataName + "_val.txt",
                transform=valid_transform,
                mode="adv",
                classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        clean_loader = data.DataLoader(
            scene_dataset(
                root_dir=args.data_root_dir,
                pathfile="./dataset/" + DataName + "_val.txt",
                transform=valid_transform,
                classname=classname,
            ),
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise NotImplementedError

    return adv_loader, clean_loader
