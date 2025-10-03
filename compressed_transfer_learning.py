import os
import time
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils import data
from torchvision import transforms


from src.utils.eval_utils import test_top5_acc, test_top5_acc_attc
from src.utils.io_utils import create_csv_files
from src.models.model_portfolio import load_model
from src.low_rank_layers.layer_utils import transform_to_low_rank
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.general_utils import get_available_device
from src.utils.attacks import get_attack
from src.utils.data_utils import get_dataset_info, get_dataloader


def main(args):
    print(args.dlrt)
    get_available_device(multi_gpu=False)

    classname, num_classes, DataName = get_dataset_info(args)

    if args.dlrt:
        save_path_prefix = (
            args.save_path_prefix
            + DataName
            + "/Pretrain/low_rank/"
            + args.network
            + "/"
        )
    else:
        save_path_prefix = (
            args.save_path_prefix
            + DataName
            + "/Pretrain/baseline/"
            + args.network
            + "/"
        )

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)

    train_loader, val_loader = get_dataloader(args, DataName, classname)

    print("Network: %s" % args.network)
    Model = load_model(args.network, num_classes)
    Model = Model.cuda()

    if args.dlrt:
        Model, lr_layers = transform_to_low_rank(
            Model, max_rank=args.rmax, init_rank=args.init_r, tol=args.tol
        )
        print("Number of low-rank layers: ", len(lr_layers))
    else:
        lr_layers = []
    print(Model)

    if args.load_model != 0:  # continue training a pre-trained low-rank model.
        model_name = args.load_model_name + ".pth"
        print(
            "Loading model from {}".format(os.path.join(save_path_prefix, model_name))
        )
        saved_state_dict = torch.load(os.path.join(save_path_prefix, model_name))
        new_params = Model.state_dict().copy()

        for i, j in zip(saved_state_dict, new_params):
            new_params[j] = saved_state_dict[i]

        Model.load_state_dict(new_params)
        print("--- Model structure ----")
        print(Model)
        print("--- Initial ranks ----")
        for layer in lr_layers:
            print(layer.r)
        print("--- Initial ranks ----")

        Model.load_state_dict(new_params)

    if args.wandb == 1:
        import wandb

        if args.dlrt:
            run_name = "{}_model_lr-{}_data-{}".format(
                args.wandb_tag,
                args.network,
                args.dataID,
            )
        else:
            run_name = "{}_model-{}_data-{}".format(
                args.wandb_tag,
                args.network,
                args.dataID,
            )

        wandb.init(project=run_name)
        wandb.config.update(args)
        wandb.watch(Model)

    Model_optimizer = torch.optim.AdamW(
        Model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Calculate the number of batches
    num_batches = len(train_loader)

    # Initialize LinearLR scheduler for model training
    scheduler = CosineAnnealingLR(
        Model_optimizer,
        T_max=(args.num_epochs + args.num_epochs_low_rank_ft) * num_batches,
    )

    num_batches = len(train_loader)

    cls_loss = torch.nn.CrossEntropyLoss()
    num_steps = args.num_epochs * num_batches
    num_steps_ft = args.num_epochs_low_rank_ft * num_batches
    hist = np.zeros((num_steps + num_steps_ft, 3))
    index_i = -1

    augmented = False

    print("Start low-rank training") if args.dlrt else print("Start baseline training")

    # ----- acceleration ----
    torch.set_float32_matmul_precision("high")

    for epoch in range(args.num_epochs + args.num_epochs_low_rank_ft):

        if epoch == args.num_epochs:
            print("Finished rank adaptive training, start finetuning")

        if (epoch) % 3 == 0 and epoch > 0:
            OA_new, top5_acc, _, _ = test_top5_acc(
                Model,
                classname,
                val_loader,
                epoch + 1,
                num_classes,
                print_per_batches=args.print_per_batches,
                is_vit=(
                    True
                    if args.network.startswith("vit")
                    or args.network.startswith("oreole")
                    else False
                ),
            )
            print(
                f"[VAL] Epoch {epoch}: Overall Accuracy = {OA_new:.4f} Top5 Accuracy = {top5_acc:.4f}"
            )

            if args.wandb == 1:
                wandb.log({"val_acc_clean": OA_new, "epoch": epoch})
                wandb.log({"val_top5_acc_clean": top5_acc, "epoch": epoch})

        for batch_index, src_data in enumerate(train_loader):

            index_i += 1

            tem_time = time.time()
            Model.train()
            Model_optimizer.zero_grad()

            X_train, Y_train, _ = src_data

            X_train = X_train.cuda()
            Y_train = Y_train.cuda()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.network.startswith("vit"):
                    output = Model(X_train).logits
                elif args.network.startswith("oreole"):
                    output = Model(X_train).logits
                else:
                    _, output = Model(X_train)
                # CE Loss
                _, src_prd_label = torch.max(output, 1)
                cls_loss_value = cls_loss(output, Y_train)
                if args.robusteness_regularization_beta > 0:
                    for layer in lr_layers:
                        cls_loss_value += layer.robustness_regularization(
                            beta=args.robusteness_regularization_beta
                        )
            cls_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)

            if args.dlrt:
                # ----- DLRT -----
                # 1) Augment
                if batch_index % args.num_local_iter == 0 and epoch < args.num_epochs:
                    augmented = True
                    # print("augment")
                    for layer in lr_layers:
                        layer.augment(Model_optimizer)
                else:
                    # 2) Train
                    for layer in lr_layers:
                        layer.set_basis_grad_zero()
                    Model_optimizer.step()  # This would be the standard training step

                # 3) Truncate
                if (
                    batch_index % args.num_local_iter == args.num_local_iter - 1
                    and epoch < args.num_epochs
                ):
                    augmented = False
                    # print("truncate")
                    for layer in lr_layers:
                        layer.truncate(Model_optimizer)
                elif (
                    augmented and batch_index == len(train_loader) - 1
                ):  # make sure not end epoch on augmented network
                    augmented = False
                    for layer in lr_layers:
                        layer.truncate(Model_optimizer)
            else:
                Model_optimizer.step()

            # ----- DLRT -----
            # Step the scheduler
            scheduler.step()
            # torch.cuda.synchronize()

            hist[index_i, 0] = time.time() - tem_time
            hist[index_i, 1] = cls_loss_value.item()
            hist[index_i, 2] = torch.mean((src_prd_label == Y_train).float()).item()

            tem_time = time.time()
            if (index_i + 1) % args.print_per_batches == 0:
                lr_params = 0
                full_params = 0
                for layer in lr_layers:
                    lr_params += layer.compute_lr_params()
                    full_params += layer.compute_dense_params()
                if full_params > 0:
                    cr = 1 - lr_params / full_params
                else:
                    cr = 0

                print(
                    "Overall Step %d; Epoch %d/%d:  %d/%d Time: %.2f cls_loss = %.3f acc = %.3f cr = %.2f time = %.2f\n"
                    % (
                        index_i,
                        epoch + 1,
                        args.num_epochs,
                        batch_index + 1,
                        num_batches,
                        np.mean(
                            hist[index_i - args.print_per_batches + 1 : index_i + 1, 0]
                        ),
                        np.mean(
                            hist[index_i - args.print_per_batches + 1 : index_i + 1, 1]
                        ),
                        np.mean(
                            hist[index_i - args.print_per_batches + 1 : index_i + 1, 2]
                        ),
                        cr,
                        hist[index_i, 0],
                    )
                )

                # print([lr_layer.r for lr_layer in lr_layers])

                if args.wandb == 1:
                    current_lr = Model_optimizer.param_groups[0]["lr"]

                    wandb.log(
                        {
                            "Time": np.mean(
                                hist[
                                    index_i - args.print_per_batches + 1 : index_i + 1,
                                    0,
                                ]
                            ),
                            "cls_loss": np.mean(
                                hist[
                                    index_i - args.print_per_batches + 1 : index_i + 1,
                                    1,
                                ]
                            ),
                            "acc": np.mean(
                                hist[
                                    index_i - args.print_per_batches + 1 : index_i + 1,
                                    2,
                                ]
                            ),
                            "compression": cr,
                            "low rank params": lr_params,
                            "full params": full_params,
                            "rank ": [lr_layer.r for lr_layer in lr_layers],
                            "learning_rate": current_lr,  # Log the learning rate
                        }
                    )

            del X_train, Y_train

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

    print("Save Model at " + os.path.join(save_path_prefix, model_name))
    torch.save(Model.state_dict(), os.path.join(save_path_prefix, model_name))

    OA_new, top5_acc, _, _ = test_top5_acc(
        Model,
        classname,
        val_loader,
        epoch + 1,
        num_classes,
        print_per_batches=args.print_per_batches,
        is_vit=(
            True
            if args.network.startswith("vit") or args.network.startswith("oreole")
            else False
        ),
    )
    if args.wandb == 1:
        wandb.log(
            {
                "val_acc_clean_final": OA_new,
                "acc*cr": OA_new * cr,
                "val_top5_acc_clean_final": top5_acc,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataID", type=int, default=1)
    parser.add_argument(
        "--network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument("--save_path_prefix", type=str, default="./models/")
    parser.add_argument("--log_path", type=str, default="./results/")
    parser.add_argument("--load_model", type=int, default=0)
    parser.add_argument("--load_model_name", type=str, default="default_model")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./dataset/data_adversarial_rs/",
        help="dataset path.",
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--print_per_batches", type=int, default=5)
    parser.add_argument("--save_name", type=str, default="default_model")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # --- low-rank parameters ----
    parser.add_argument("--dlrt", type=int, default=0)
    parser.add_argument("--num_epochs_low_rank_ft", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.075)
    parser.add_argument("--rmax", type=float, default=200)
    parser.add_argument("--init_r", type=float, default=50)
    parser.add_argument("--num_local_iter", type=int, default=10)
    parser.add_argument("--robusteness_regularization_beta", type=float, default=0.0)

    # ---- robustness regularization parameters ----

    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument("--git", type=float, default=0.05)
    parser.add_argument("--wandb_tag", type=str, default="model")

    main(parser.parse_args())
