import os
import argparse
import torch
from src.utils.eval_utils import test_top5_acc
from src.low_rank_layers.layer_utils import transform_to_low_rank
from src.utils.data_utils import get_dataset_info, get_adv_valloader
from src.models.model_portfolio import load_model


def main(args):
    classname, num_classes, DataName = get_dataset_info(args)

    if args.dlrt:

        adv_root_dir = (
            args.data_root_dir
            + DataName
            + "_adv/"
            + args.attack_func
            + "/low_rank/"
            + args.surrogate_network
            + "/"
        )
    else:
        adv_root_dir = (
            args.data_root_dir
            + DataName
            + "_adv/"
            + args.attack_func
            + "/baseline/"
            + args.surrogate_network
            + "/"
        )
    args.root_dir = args.data_root_dir

    adv_loader, clean_loader = get_adv_valloader(
        args, DataName, classname, adv_root_dir
    )

    print("Network: %s" % args.target_network)
    Model = load_model(args.target_network, num_classes)
    Model = Model.cuda()

    # Transform model to low-rank factorization

    if args.dlrt:
        Model, lr_layers = transform_to_low_rank(
            Model, max_rank=args.rmax, init_rank=args.init_r, tol=args.tol
        )
        print("Number of low-rank layers: ", len(lr_layers))
    else:
        lr_layers = []

    # print(Model)

    descriptor = "low_rank" if args.dlrt else "baseline"

    dirpath = (
        args.save_path_prefix
        + DataName
        + "/Pretrain/"
        + descriptor
        + "/"
        + args.target_network
        + "/"
    )

    if args.dlrt:
        model_name = (
            "beta_"
            + str(args.robusteness_regularization_beta)
            + "_tol_"
            + str(args.tol)
            + "_rmax_"
            + str(args.rmax)
            + "_init_rank_"
            + str(args.init_r)
            + ".pth"
        )
    else:
        model_name = "beta_" + str(args.robusteness_regularization_beta) + ".pth"

    model_path_resume = os.path.join(dirpath, model_name)
    saved_state_dict = torch.load(model_path_resume)

    new_params = Model.state_dict().copy()

    for i, j in zip(saved_state_dict, new_params):
        new_params[j] = saved_state_dict[i]

    Model.load_state_dict(new_params)

    if args.wandb == 1:
        import wandb

        project_name = "{}-target{}_surrogate{}_data-{}".format(
            args.wandb_tag,
            args.target_network,
            args.surrogate_network,
            args.dataID,
        )
        project_name = "lr_" + project_name if args.dlrt else project_name
        wandb.init(project=project_name)

        wandb.config.update(args)
        wandb.watch(Model)

    Model.eval()
    OA_adv, top5_acc, class_acc, class_names = test_top5_acc(
        Model,
        classname,
        adv_loader,
        1,
        num_classes,
        print_per_batches=10,
        is_vit="vit" in args.target_network or "oreole" in args.target_network,
    )
    OA_clean, clean_top5_acc, clean_class_acc, clean_class_names = test_top5_acc(
        Model,
        classname,
        clean_loader,
        1,
        num_classes,
        print_per_batches=10,
        is_vit="vit" in args.target_network or "oreole" in args.target_network,
    )
    print("Clean Test Set OA:", OA_clean * 100)
    print(args.attack_func + " Test Set OA:", OA_adv * 100)

    if args.wandb == 1:

        wandb.log(
            {
                "clean_val_acc": OA_clean,
                "adv_val_acc_" + args.attack_func: OA_adv,
                "clean_top5_acc": clean_top5_acc,
                "adv_top5_acc_" + args.attack_func: top5_acc,
            }
        )
        for name, acc in zip(class_names, class_acc):
            wandb.log({"class_" + name: acc})

        for name, acc in zip(clean_class_names, clean_class_acc):
            wandb.log({"clean_class_" + name: acc})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataID", type=int, default=1)

    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="./dataset/data_adversarial_rs/",
        help="data path for imagenet",
    )
    parser.add_argument(
        "--target_network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument(
        "--surrogate_network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument("--save_path_prefix", type=str, default="./models/")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--attack_func",
        type=str,
        default="fgsm",
        help="fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut",
    )
    # Low rank parameters
    parser.add_argument("--dlrt", type=int, default=0)
    parser.add_argument(
        "--rmax", type=float, default=200
    )  # this must match the max rank of the trained model (Needs to be made nicer later)
    parser.add_argument("--tol", type=float, default=0.075)
    parser.add_argument("--robusteness_regularization_beta", type=float, default=0.05)
    parser.add_argument("--init_r", type=float, default=50)
    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument(
        "--attack_epsilon", type=float, default=1
    )  # only for wandb registry
    parser.add_argument("--wandb_tag", type=str, default="model")

    main(parser.parse_args())
