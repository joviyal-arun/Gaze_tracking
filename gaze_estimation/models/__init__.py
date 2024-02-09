import importlib
import timm
import torch
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    dataset_name = config.mode.lower()
    print("NAME :-",dataset_name,"NAME2 :-",config.model.name)
    if dataset_name == 'eth_xgaze':
        model_name='resnet18'
        model = timm.create_model(model_name, num_classes=2)
    else:
        module = importlib.import_module(
            f'gaze_estimation.models.{dataset_name}.{config.model.name}')
        model = module.Model(config)
    device = torch.device(config.device)
    model.to(device)
    return model
