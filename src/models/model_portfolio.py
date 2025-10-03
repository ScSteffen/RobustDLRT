import torch
from torch import nn
from src.models import model, model_vit


def load_model(model_name, num_classes):
    if model_name == "alexnet":
        Model = model.alexnet(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "vgg11":
        Model = model.vgg11(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "vgg16":
        Model = model.vgg16(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "vgg19":
        Model = model.vgg19(pretrained=True)
        Model.classifier._modules["6"] = nn.Linear(4096, num_classes)
    elif model_name == "inception":
        Model = model.inception_v3(pretrained=True, aux_logits=False)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet18":
        Model = model.resnet18(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        Model = model.resnet34(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        Model = model.resnet50(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        Model = model.resnet101(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnext50_32x4d":
        Model = model.resnext50_32x4d(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "resnext101_32x8d":
        Model = model.resnext101_32x8d(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        Model = model.densenet121(pretrained=True)
        Model.classifier = nn.Linear(1024, num_classes)
    elif model_name == "densenet169":
        Model = model.densenet169(pretrained=True)
        Model.classifier = nn.Linear(1664, num_classes)
    elif model_name == "densenet201":
        Model = model.densenet201(pretrained=True)
        Model.classifier = nn.Linear(1920, num_classes)
    elif model_name == "regnet_x_400mf":
        Model = model.regnet_x_400mf(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "regnet_x_8gf":
        Model = model.regnet_x_8gf(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "regnet_x_16gf":
        Model = model.regnet_x_16gf(pretrained=True)
        Model.fc = torch.nn.Linear(Model.fc.in_features, num_classes)
    elif model_name == "vit16b":
        Model = model_vit.vit_b16(num_classes)
    elif model_name == "vit16l":
        Model = model_vit.vit_l16(num_classes)
    elif model_name == "vit32l":
        Model = model_vit.vit_l32(num_classes)
    elif model_name == "vit14h":
        Model = model_vit.vit_h14(num_classes)
    elif model_name == "vit14g":
        Model = model_vit.vit_g14(num_classes)
    elif model_name == "oreole":
        # Huggingface Transformer setup
        from transformers import ViTForImageClassification

        Model = ViTForImageClassification.from_pretrained(
            "hf_vitB_from_oreole",
            torch_dtype=torch.float32,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    elif model_name == "oreoleB":
        from transformers import ViTForImageClassification

        Model = ViTForImageClassification.from_pretrained(
            "hf_vitB_from_oreole",
            torch_dtype=torch.float32,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    elif model_name == "oreoleH":
        from transformers import ViTForImageClassification

        Model = ViTForImageClassification.from_pretrained(
            "hf_vitH_from_oreole",
            torch_dtype=torch.float32,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    elif model_name == "oreoleG":
        from transformers import ViTForImageClassification

        Model = ViTForImageClassification.from_pretrained(
            "hf_vitG_from_oreole",
            torch_dtype=torch.float32,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    return Model
