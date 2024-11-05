import torch.optim as optim
import argparse
import subprocess
import os

from src.object_detection.data_handler.data_utils import choose_dataset

from src.object_detection.train import Trainer
import torch

force_cpu = False  # Forces the model to train on CPU. For debug purposes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script with hyperparameter settings"
    )
    # Benchmark
    parser.add_argument(
        "--benchmark",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="1=Cifar10, 2=Cifar100, 3=Tiny-Imagenet, 4=Imagenet",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="1=VGG16, 2=AlexNet, 3=ResNet18, 4=ViT-small, 5=ViT,  6=ViT_16",
    )
    parser.add_argument(
        "--pretrained", type=int, default=1, choices=[0, 1], help="0=No, 1=Yes"
    )
    # Hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )

    parser.add_argument(
        "--lr_integrator_choice",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Integrator choice (1=Dense, 2=BUG, 3=not yet defined,  4=Simultaneous descend)",
    )

    parser.add_argument(
        "--initial_cr", type=float, default=0.8, help="Initial compression ratio"
    )

    parser.add_argument(
        "--max_rank", type=int, default=300, help="Maximum rank for compression"
    )

    parser.add_argument(
        "--tol", type=float, default=0.1, help="Tolerance for compression"
    )

    parser.add_argument(
        "--num_local_iter",
        type=int,
        default=20,
        help="Number of coefficient steps per low-rank update",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=5e-3, help="Learning rate for training"
    )

    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs for training"
    )

    parser.add_argument(
        "--momentum", type=float, default=0.3, help="Momentum for training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Wegith decay for training"
    )

    # Output file
    parser.add_argument(
        "--output", type=str, default="output.csv", help="Output file path"
    )

    parser.add_argument(
        "--wandb", type=int, default=0, help="Activate wandb logging: 0=no, 1=yes"
    )

    parser.add_argument(
        "--load_checkpoint", type=int, default=0, help="Load checkpoint"
    )

    args = parser.parse_args()

    return args


def print_arg_choices(args):
    if args.benchmark == 1:
        benchmark = "Cifar10"
    elif args.benchmark == 2:
        benchmark = "Cifar100"
    elif args.benchmark == 3:
        benchmark = "TinyImagenet"
    elif args.benchmark == 4:
        benchmark = "Imagenet"
    else:
        print("Model not defined")
        exit(1)
    print(f"  Benchmark: {benchmark}")

    if args.model == 1:
        model = "VGG16"
    elif args.model == 2:
        model = "AlexNet"
    elif args.model == 3:
        model = "ResNet18"
    elif args.model == 4:
        model = "ViT-small"
    elif args.model == 5:
        model = "ViT"
    elif args.model == 6:
        model = "vit_b_16"
    else:
        print("Model not defined")
        exit(1)
    print(f"  Benchmarking Model: {model}")
    if args.pretrained == 0:
        print("Model uses randomly initialized weights")
    else:
        print("Model uses pretrained weights")

    if args.lr_integrator_choice == 1:
        integrator = "Dense"
    elif args.lr_integrator_choice == 2:
        integrator = "BUG"
    elif args.lr_integrator_choice == 3:
        integrator = "Parallel"
    elif args.lr_integrator_choice == 4:
        integrator = "Simulataneous Descend"
    else:
        print("Integrator not defined")
        exit(1)

    print(f"  Integrator Choice: {integrator}")

    print("Low Rank Settings:")
    print(f"  Initial Compression Ratio: {args.initial_cr}")
    print(f"  Maximum Rank for Compression: {args.max_rank}")
    print(f"  Tolerance for Compression: {args.tol}")
    print(f"  Number of Coefficient steps per Low-Rank Update: {args.num_local_iter}")

    print("\nTraining Hyperparameters:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Number of Epochs: {args.epochs}")

    print("\n Log Output:")
    print(f"  File Location: {args.output}")
    print(f"  Wandb activated?: {bool(args.wandb)}")


def choose_model(args, device):
    if args.benchmark == 1:  # cifar10
        num_classes = 10
    elif args.benchmark == 2:  # cifar100
        num_classes = 100
    elif args.benchmark == 3:  # tiny imagenet
        num_classes = 200
    elif args.benchmark == 4:  # imagenet
        num_classes = 1000

    lr_layers_list = []
    # Initialize the model
    if args.model == 1:
        if args.lr_integrator_choice == 2:
            from src.object_detection.vgg_convlr import (
                vgg16,
                set_lr_options,
                lr_layers,
            )  # Convolutions are also low-rank in this version.

            set_lr_options(
                ranks=[args.max_rank, args.max_rank],
                init_cr=args.initial_cr,
                tol=args.tol,
                rmin=num_classes,
                lr_method=args.lr_integrator_choice,
                device=device,
                benchmark=args.benchmark,
            )
            model = vgg16(num_classes=num_classes).to(device)
            lr_layers_list = model.lr_layers
            # print(lr_layers_list)

            # Print all named parameters
            # for name, param in model.named_parameters():
            #    print(f"Name: {name}, Shape: {param.shape}")

        elif args.lr_integrator_choice == 1:
            from src.object_detection.vgg_conv_reference import vgg16, set_lr_options

            set_lr_options(
                ranks=[args.max_rank, args.max_rank],
                init_cr=args.initial_cr,
                tol=args.tol,
                rmin=num_classes,
                lr_method=args.lr_integrator_choice,
                device=device,
                benchmark=args.benchmark,
            )
            model = vgg16(num_classes=num_classes).to(device)

    elif args.model == 2:
        from src.object_detection.alexnet import set_lr_options, alexnet, lr_layers

        set_lr_options(
            ranks=[args.max_rank, args.max_rank],
            init_cr=args.initial_cr,
            tol=args.tol,
            rmin=num_classes,
            lr_method=args.lr_integrator_choice,
            device=device,
            benchmark=args.benchmark,
        )
        model = alexnet(num_classes=num_classes).to(device)
        lr_layers_list = lr_layers
    elif args.model == 3:  # Resnet
        if args.pretrained == 1:  # only for imagenet
            from src.object_detection.resnets.pretrained_resnet import (
                set_lr_options,
                PreTrainedResNet18,
                lr_layers,
            )

            set_lr_options(
                ranks=args.max_rank,
                init_cr=args.initial_cr,
                tol=args.tol,
                rmin=num_classes,
                lr_method=args.lr_integrator_choice,
                device=device,
                benchmark=args.benchmark,
            )

            model = PreTrainedResNet18(num_classes=num_classes, pretrained=True).to(
                device
            )
            lr_layers_list = lr_layers
            model.to(device)
        else:
            from src.object_detection.resnets.resnet_pytorch import (
                set_lr_options,
                resnet18,
                lr_layers,
            )

            set_lr_options(
                ranks=args.max_rank,
                init_cr=args.initial_cr,
                tol=args.tol,
                rmin=num_classes,
                lr_method=args.lr_integrator_choice,
                device=device,
                benchmark=args.benchmark,
            )
            model = resnet18(num_classes=num_classes).to(device)
            lr_layers_list = lr_layers
        """
        from torchvision.models import resnet18, resnet50, wide_resnet50_2

        model = resnet18(pretrained=False, num_classes=num_classes).to(device)
        fc_in_shape = model.fc.in_features

        model.fc = torch.nn.Sequential(
            DenseLayer(fc_in_shape, fc_in_shape),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2),
            DenseLayer(fc_in_shape, num_classes),
        ).to(device)
        """
    elif args.model == 4:
        from src.object_detection.vit_small import ViT, set_lr_options, lr_layers

        lr_attn = True
        set_lr_options(
            rmax=args.max_rank,
            init_cr=args.initial_cr,
            tol=args.tol,
            rmin=num_classes,
            lr_method=args.lr_integrator_choice,
            device=device,
            benchmark=args.benchmark,
            low_rank_attn=lr_attn,
        )
        size = 32
        patchsize = 8
        dimhead = 512
        model = ViT(
            image_size=size,
            patch_size=patchsize,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,
            heads=2,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        ).to(device)
        lr_layers_list = lr_layers

    elif args.model == 5:
        from src.object_detection.vit import ViT

        size = 32
        patchsize = 4
        dimhead = 512
        model = ViT(
            image_size=size,
            patch_size=patchsize,
            num_classes=10,
            dim=dimhead,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        ).to(device)

    elif args.model == 6:
        from src.object_detection.vit_16 import ViT_16

        model = ViT_16(
            num_classes=num_classes, pretrained=True, freeze_attn=False, device=device
        ).vit.to(device)

    # Define the loss optimizer

    # optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,           weight_decay=args.weight_decay)
    # Define LR Scheduler
    max_epochs = args.epochs
    if args.model < 3:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_epochs, eta_min=1e-5
        )
    elif args.model == 3:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        print("Use CosineAnnealing LR Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_epochs, eta_min=5e-4
        )
    elif args.model == 6:
        optimizer = optim.Adam(
            model.parameters(),
            betas=(0.9, 0.999),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_epochs, eta_min=1e-5
        )
        print("Use CosineAnnealing LR scheduler and Adam optimizer")
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_epochs, eta_min=1e-5
        )
        print("Use CosineAnnealing LR scheduler and Adam optimizer")

    return model, optimizer, scheduler, lr_layers_list


def setup_trainer(args, device):
    model, optimizer, scheduler, lr_layers = choose_model(args, device)

    # -------- Dataset Selection -----------
    train_loader, val_loader, test_loader = choose_dataset(
        dataset_choice=args.benchmark,
        batch_size=args.batch_size,
        datapath="./data/imagenet/tmp/",
    )

    t = Trainer(
        model,
        train_loader=train_loader,
        test_loader=val_loader,
        lr_mode=args.lr_integrator_choice,
        output=args.output,
        device=device,
        scheduler=scheduler,
        optimizer=optimizer,
        model_choice=args.model,
        wandb=bool(args.wandb),
        lr_layers=lr_layers,
    )

    return t


def main():
    # Set the device (GPU or CPU)
    if torch.cuda.is_available():
        device = get_available_device()
        print(f"Using Cuda GPU {device}")
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and not force_cpu:
    #     device = torch.device("mps")
    #     print('Using Apple Silicon MPS')
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print("Start Cifar10 Benchmark using " + str(device) + " hardware.")

    args = parse_args()
    print_arg_choices(args)

    t = setup_trainer(args, device)

    ##### setting file paths... #
    path = "./results/"
    if args.benchmark == 1:
        path = path + "Cifar10/"
    elif args.benchmark == 2:
        path = path + "Cifar100/"
    # model
    if args.model == 1:
        path = path + "VGG16"
    if args.model == 2:
        path = path + "AlexNet"
    if args.model == 3:
        path = path + "ResNet"
    if args.model == 4:
        path = path + "ViT"
    path += "/saved_model/"
    ##### setting file paths... #

    if args.lr_integrator_choice == 1:
        path += "model.pth"
        path_final = path + "model_final.pth"

    if args.lr_integrator_choice == 2:
        path += "model_lr.pth"
        path_final = path + "lr_final.pth"
    if args.load_checkpoint > 0:

        if os.path.isfile(path):
            t.model.load_state_dict(torch.load(path))
            print("Loaded checkpoint model from", path)
        else:
            print("Model not found, starting pretraining")

    t.train(
        num_epochs=args.epochs,
        num_local_iter=args.num_local_iter,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        args=args,
    )
    print("Saving model to", path_final)
    torch.save(t.model.state_dict(), path_final)

    return 0


def get_available_device():
    # Get GPU memory usage using nvidia-smi
    cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    memory_used = subprocess.check_output(cmd.split()).decode().strip().split("\n")
    memory_used = [int(memory.strip()) for memory in memory_used]

    # Find GPU with least memory usage
    device = memory_used.index(min(memory_used))
    return torch.device(f"cuda:{device}")


if __name__ == "__main__":
    main()
