import torch
from torchvision import transforms


def get_attack(attack_name):
    if attack_name == "fgsm":
        return FGSM
    elif attack_name == "jitter":
        return Jitter()
    elif attack_name == "cw":
        return CW()
    else:
        raise ValueError("Unknown attack name")


def FGSM(X, Y, model, is_vit, epsilon, cls_loss):
    # model.eval()
    adv_im = X.clone()
    adv_im.requires_grad = True
    if is_vit:
        out = model(adv_im).logits
    else:
        _, out = model(adv_im)

    pred_loss = cls_loss(out, Y)
    grad = torch.autograd.grad(
        pred_loss, adv_im, retain_graph=False, create_graph=False
    )[0]

    infty_norm = torch.norm(
        grad,
        float("inf"),
        dim=[1, 2, 3],
        keepdim=True,
    )

    adv_im = adv_im + epsilon * grad
    # print(adv_im.shape)
    adv_im = adv_im / infty_norm
    delta = torch.clamp(adv_im - X, min=-epsilon, max=epsilon)
    adv_im = X + delta
    # model.train()

    # Transform back
    mean = torch.tensor([0.485, 0.456, 0.406], device=adv_im.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=adv_im.device).view(1, 3, 1, 1)

    # Unnormalize
    imgs = adv_im.detach() * std + mean

    # Clamp to [0,1] and convert to [0,255]
    imgs = torch.clamp(imgs, 0, 1) * 255
    imgs = torch.round(imgs).to(torch.uint8).to(adv_im.dtype)

    adv_im = (imgs - mean) / std

    return adv_im


def Jitter(X, Y, model, is_vit, epsilon, cls_loss, num_classes):
    pass


def PGD(X, Y, model, is_vit):
    raise NotImplementedError


def CW(X, Y, model, is_vit):
    raise NotImplementedError
