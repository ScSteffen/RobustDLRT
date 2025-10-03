import os
import argparse

from torch.nn import functional as F
from PIL import Image
from torch import nn
from torch.utils import data
from tqdm import tqdm

from src.utils.eval_utils import *
from src.models.model_portfolio import load_model
from dataset.scene_dataset import *
from src.low_rank_layers.layer_utils import transform_to_low_rank
from src.utils import imagenet_utils


def main(args):
    if args.dataID == 1:
        DataName = "UCM"
        num_classes = 21
        if args.attack_func[:2] == "mi":
            mix_file = "./dataset/UCM_" + args.attack_func + "_sample.png"
    elif args.dataID == 2:
        DataName = "AID"
        num_classes = 30
        if args.attack_func[:2] == "mi":
            mix_file = "./dataset/AID_" + args.attack_func + "_sample.png"
    elif args.dataID == 3:
        DataName = "Cifar10"
        num_classes = 10
        if args.attack_func[:2] == "mi":
            mix_file = "./dataset/Cifar10_" + args.attack_func + "_sample.png"
    elif args.dataID == 4:
        DataName = "MillionAID"
        classname = sorted(
            os.listdir(
                os.path.join(
                    os.path.join(args.data_root_dir, "/MillionAID/images"), "train"
                )
            )
        )
        num_classes = len(classname)
    elif args.dataID == 5:
        DataName = "NWPU"
        classname = sorted(
            os.listdir(
                os.path.join(os.path.join(args.data_root_dir, "/NWPU/images"), "train")
            )
        )
        num_classes = len(classname)
    elif args.dataID == 6:
        DataName = "ImageNet"
        num_classes = 1000
        classname = imagenet_utils.get_imagenet_classnames(args.data_root_dir)

        if args.attack_func[:2] == "mi":
            mix_file = (
                "./dataset/data_adversarial_rs/imagenet_"
                + args.attack_func
                + "_sample.png"
            )
    else:
        raise NotImplementedError

    if args.dlrt:
        save_path_prefix = (
            args.save_path_prefix
            + DataName
            + "_adv/"
            + args.attack_func
            + "/low_rank/"
            + args.network
            + "/"
        )
    else:
        save_path_prefix = (
            args.save_path_prefix
            + DataName
            + "_adv/"
            + args.attack_func
            + "/baseline/"
            + args.network
            + "/"
        )

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)

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
        imloader = data.DataLoader(
            scene_dataset(
                root_dir=args.save_path_prefix,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
                classname="",
            ),
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    elif args.dataID == 3:
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        composed_transforms = transforms.Compose(
            [
                transforms.Resize(size=(args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ]
        )

        imloader = data.DataLoader(
            scene_dataset(
                root_dir=save_path_prefix,
                pathfile="./dataset/" + DataName + "_test.txt",
                transform=composed_transforms,
                classname="",
                mode="adv",
            ),
            batch_size=1,
            shuffle=True,
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

        imloader = data.DataLoader(
            scene_dataset(
                root_dir=args.data_root_dir,
                pathfile="./dataset/" + DataName + "_val.txt",
                transform=composed_transforms,
                classname=classname,
            ),
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=True,
        )
    else:
        raise NotImplementedError

    ###################Network Definition###################
    print("Network: %s" % args.network)
    Model = load_model(args.network, num_classes)

    # Transform model to low-rank factorization
    attack_model = Model.cuda()

    if args.dlrt:
        attack_model, _ = transform_to_low_rank(
            attack_model, max_rank=args.rmax, init_rank=10, tol=args.tol
        )

        surrogate_model_path = (
            args.model_root_dir + DataName + "/Pretrain/low_rank/" + args.network + "/"
        )

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
        surrogate_model_path = (
            args.model_root_dir + DataName + "/Pretrain/baseline/" + args.network + "/"
        )
        model_name = "beta_" + str(args.robusteness_regularization_beta) + ".pth"

    model_path_resume = os.path.join(surrogate_model_path, model_name)

    print("load model from")
    print(model_path_resume)
    saved_state_dict = torch.load(model_path_resume)
    new_params = attack_model.state_dict().copy()

    for i, j in zip(saved_state_dict, new_params):
        new_params[j] = saved_state_dict[i]

    attack_model.load_state_dict(new_params)

    attack_model = attack_model.cuda()
    attack_model.eval()

    num_batches = len(imloader)

    kl_loss = torch.nn.KLDivLoss()
    cls_loss = torch.nn.CrossEntropyLoss()
    tpgd_loss = torch.nn.KLDivLoss(reduction="sum")
    mse_loss = torch.nn.MSELoss(reduction="none")
    tbar = tqdm(imloader)

    num_iter = 5
    is_vit = (
        True
        if args.network.startswith("vit") or args.network.startswith("oreole")
        else False
    )

    torch.set_float32_matmul_precision("high")

    if args.attack_func == "fgsm":
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data

            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            tbar.set_description("Batch: %d/%d" % (batch_index + 1, num_batches))
            adv_im.requires_grad = True

            pred_loss = 0
            for k in range(args.SIM):
                if is_vit:
                    out = attack_model(adv_im / (2 ** (k))).logits
                else:
                    _, out = attack_model(adv_im / (2 ** (k)))
                pred_loss += cls_loss(out, label)
            grad = (1 / args.SIM) * torch.autograd.grad(
                pred_loss, adv_im, retain_graph=False, create_graph=False
            )[0]

            adv_im = adv_im.detach() + args.attack_epsilon * grad / torch.norm(
                grad, float("inf")
            )
            delta = torch.clamp(
                adv_im - X, min=-args.attack_epsilon, max=args.attack_epsilon
            )
            adv_im = (X + delta).detach()

            recreated_image = recreate_image(adv_im.cpu())

            gen_name = img_name[0] + "_adv.png"
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")

    elif args.attack_func == "condlr_fgsm":
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            tbar.set_description("Batch: %d/%d" % (batch_index + 1, num_batches))
            adv_im.requires_grad = True
            max_val = torch.max(adv_im)
            min_val = torch.min(adv_im)

            pred_loss = 0
            for k in range(args.SIM):
                if is_vit:
                    out = attack_model(adv_im / (2 ** (k))).logits
                else:
                    _, out = attack_model(adv_im / (2 ** (k)))
                pred_loss += cls_loss(out, label)
            grad = (1 / args.SIM) * torch.autograd.grad(
                pred_loss, adv_im, retain_graph=False, create_graph=False
            )[0]

            # std Taken from Emanueles paper (for cifar10 only!).
            std = torch.tensor([0.2023, 0.1994, 0.2010], device=adv_im.device).view(
                1, 3, 1, 1
            )  # Ensure correct shape and device

            adv_im = torch.clamp(
                adv_im.detach() + args.attack_epsilon * (grad.sign() / std),
                min_val,
                max_val,
            )
            recreated_image = recreate_image(adv_im.cpu())

            gen_name = img_name[0] + "_adv.png"
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")

    elif args.attack_func == "ifgsm":
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description(
                    "Batch: %d/%d, Iteration:%d" % (batch_index + 1, num_batches, i + 1)
                )
                adv_im.requires_grad = True

                pred_loss = 0
                for k in range(args.SIM):
                    if is_vit:
                        out = attack_model(adv_im / (2 ** (k))).logits
                    else:
                        _, out = attack_model(adv_im / (2 ** (k)))
                    pred_loss += cls_loss(out, label)
                grad = (1 / args.SIM) * torch.autograd.grad(
                    pred_loss, adv_im, retain_graph=False, create_graph=False
                )[0]

                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float("inf"))
                delta = torch.clamp(
                    adv_im - X, min=-args.attack_epsilon, max=args.attack_epsilon
                )
                adv_im = (X + delta).detach()

                if args.dataID == 3:
                    adv_im = adv_im.resize((32, 32), resample=Image.BILINEAR)

                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + "_adv.png"
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")

    elif args.attack_func == "tpgd":
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            Y = Y.numpy().squeeze()

            if args.network == "vit32b":
                logit_ori = attack_model(X).logits
            else:
                _, logit_ori = attack_model(X)

            logit_ori = logit_ori.detach()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description(
                    "Batch: %d/%d, Iteration:%d" % (batch_index + 1, num_batches, i + 1)
                )
                adv_im.requires_grad = True

                if args.network == "vit32b":
                    logit_adv = attack_model(adv_im).logits
                else:
                    _, logit_adv = attack_model(adv_im)

                pred_loss = tpgd_loss(
                    F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1)
                )

                grad = torch.autograd.grad(
                    pred_loss, adv_im, retain_graph=False, create_graph=False
                )[0]

                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float("inf"))
                delta = torch.clamp(
                    adv_im - X, min=-args.attack_epsilon, max=args.attack_epsilon
                )
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + "_adv.png"
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")

    elif args.attack_func == "jitter":
        m = args.SIM
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            label_mask = F.one_hot(Y, num_classes=num_classes).cuda().float()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()

            for i in range(num_iter):
                tbar.set_description(
                    "Batch: %d/%d, Iteration:%d" % (batch_index + 1, num_batches, i + 1)
                )
                adv_im.requires_grad = True
                sum_loss = 0
                for k in range(m):
                    if is_vit:
                        out = attack_model(adv_im).logits
                    else:
                        _, out = attack_model(adv_im)

                    _, pre = torch.max(out, dim=1)
                    wrong = pre != label

                    norm_z = torch.norm(out, p=float("inf"), dim=1, keepdim=True)
                    hat_z = nn.Softmax(dim=1)(args.scale * out / norm_z)

                    hat_z = hat_z + args.std * torch.randn_like(hat_z)

                    loss = mse_loss(hat_z, label_mask).mean(dim=1)

                    norm_r = torch.norm((adv_im - X), p=float("inf"), dim=[1, 2, 3])
                    nonzero_r = norm_r != 0
                    loss[wrong * nonzero_r] /= norm_r[wrong * nonzero_r]
                    loss = loss.mean()
                    sum_loss += loss
                grad = torch.autograd.grad(
                    sum_loss / m, adv_im, retain_graph=False, create_graph=False
                )[0]

                adv_im = adv_im.detach() + args.attack_epsilon * grad / torch.norm(
                    grad, float("inf")
                )
                delta = torch.clamp(
                    adv_im - X, min=-args.attack_epsilon, max=args.attack_epsilon
                )
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                # Process confirmation image
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + "_adv.png"
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")

    elif args.attack_func == "mixup" or args.attack_func == "mixcut":
        mixup_im = (
            composed_transforms(Image.open(mix_file).convert("RGB")).unsqueeze(0).cuda()
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if is_vit:
                mixup_feature = attack_model(mixup_im).logits
            else:
                mixup_feature, _ = attack_model(mixup_im)

        mixup_feature = mixup_feature.data
        for batch_index, src_data in enumerate(tbar):

            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()
            momentum = torch.zeros_like(X).cuda()

            # Start iteration
            for i in range(num_iter):
                tbar.set_description(
                    "Batch: %d/%d, Iteration:%d" % (batch_index + 1, num_batches, i + 1)
                )
                adv_im.requires_grad = True
                pred_loss = 0
                mix_loss = 0
                for k in range(args.SIM):
                    # Scale augmentation
                    if is_vit:
                        out = attack_model(adv_im / (2 ** (k))).logits
                        feature = out.clone()
                    else:
                        feature, out = attack_model(adv_im / (2 ** (k)))

                    pred_loss += cls_loss(out, label)
                    mix_loss += -kl_loss(feature, mixup_feature)

                total = pred_loss * args.beta + mix_loss
                grad = torch.autograd.grad(
                    total, adv_im, retain_graph=False, create_graph=False
                )[0]

                grad = grad / torch.norm(grad, p=1)
                grad = grad + momentum * args.decay
                momentum = grad
                alpha = 1.0
                adv_im = adv_im.detach() + alpha * grad / torch.norm(grad, float("inf"))
                delta = torch.clamp(
                    adv_im - X, min=-args.attack_epsilon, max=args.attack_epsilon
                )
                adv_im = (X + delta).detach()

                recreated_image = recreate_image(adv_im.cpu())
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + "_adv.png"

            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")
    elif (
        args.attack_func == "nifgsm"
    ):  # implement NI-FGSM from ref paper Algorithm 1 (Lin et al 2020)
        alpha = args.attack_epsilon / num_iter  # learning rate
        m = args.SIM
        for batch_index, src_data in enumerate(tbar):
            X, Y, img_name = src_data
            X = X.cuda()
            adv_im = X.clone().cuda()
            label = Y.clone().cuda()
            Y = Y.numpy().squeeze()
            # begin attack iterations
            grad_i = torch.zeros_like(X).cuda()
            for i in range(num_iter):
                tbar.set_description(
                    "Batch: %d/%d, Iteration:%d" % (batch_index + 1, num_batches, i + 1)
                )
                adv_im.requires_grad = True
                # begin SIM iterations
                loss = 0
                nes_grad = torch.zeros_like(X).cuda()
                nes_im = adv_im + alpha * args.decay * grad_i
                for k in range(m):
                    if args.attack_network == "vit32b":
                        pred = attack_model(nes_im / (2 ** (k))).logits
                    else:
                        _, pred = attack_model(nes_im / (2 ** (k)))
                    loss += cls_loss(pred, label)
                nes_grad = torch.autograd.grad(
                    loss / m, nes_im, retain_graph=False, create_graph=False
                )[0]
                grad_i = args.decay * grad_i + nes_grad / torch.norm(nes_grad, p=1)

                adv_im = adv_im.detach() + alpha * torch.sign(grad_i)
                delta = torch.clamp(
                    adv_im - X, min=-args.attack_epsilon, max=args.attack_epsilon
                )
                adv_im = (X + delta).detach()
                recreated_image = recreate_image(adv_im.cpu())
                adv_im = preprocess_image(Image.fromarray(recreated_image), args)

            gen_name = img_name[0] + "_adv.png"
            im = Image.fromarray(recreated_image)
            im.save(save_path_prefix + gen_name, "png")

    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataID", type=int, default=1)
    parser.add_argument(
        "--network",
        type=str,
        default="vgg16",
        help="alexnet,vgg11,vgg16,vgg19,inception,resnet18,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d,densenet121,densenet169,densenet201,regnet_x_400mf,regnet_x_8gf,regnet_x_16gf",
    )
    parser.add_argument(
        "--save_path_prefix", type=str, default="./dataset/data_adversarial_rs/"
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="./dataset/data_adversarial_rs/",
        help="data path.",
    )
    parser.add_argument(
        "--model_root_dir",
        type=str,
        default="./models/",
        help="pre-trained model path.",
    )
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--attack_func",
        type=str,
        default="fgsm",
        help="fgsm,ifgsm,cw,tpgd,jitter,mixup,mixcut",
    )
    parser.add_argument("--attack_epsilon", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument("--decay", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--scale", type=float, default=10)
    parser.add_argument("--std", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=50)

    # Low rank parameters
    parser.add_argument("--dlrt", type=int, default=0)
    parser.add_argument(
        "--rmax", type=float, default=200
    )  # this must match the max rank of the trained model (Needs to be made nicer later)
    parser.add_argument("--tol", type=float, default=0.075)
    parser.add_argument("--robusteness_regularization_beta", type=float, default=0.05)
    parser.add_argument("--init_r", type=float, default=50)
    parser.add_argument("--SIM", type=int, default=1)

    main(parser.parse_args())
