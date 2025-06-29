import torchvision
import torchinfo
from torch import nn
print("inception")
inception = torchvision.models.inception_v3(pretrained=True)
modules = list(inception.children())[:-3]
inception = nn.Sequential(*modules)
print(torchinfo.summary(inception, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
#
print("resnet152")
resnet152 = torchvision.models.resnet152(weights="IMAGENET1K_V2")
modules = list(resnet152.children())[:-2]
resnet152 = nn.Sequential(*modules)
print(torchinfo.summary(resnet152, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("resnet101")
resnet101 = torchvision.models.resnet101(pretrained=True)
modules = list(resnet101.children())[:-2]
resnet101 = nn.Sequential(*modules)
print(torchinfo.summary(resnet101, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))

print("dense161")
densenet161 = torchvision.models.densenet161(pretrained=True)
modules = list(densenet161.children())[:-1]
densenet161 = nn.Sequential(*modules)
print(torchinfo.summary(densenet161, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("dense121")
densenet121 = torchvision.models.densenet121(pretrained=True)
modules = list(densenet121.children())[:-1]
densenet121 = nn.Sequential(*modules)
print(torchinfo.summary(densenet121, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("dense201")
densenet201 = torchvision.models.densenet201(pretrained=True)
modules = list(densenet201.children())[:-1]
densenet201 = nn.Sequential(*modules)
print(torchinfo.summary(densenet201, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("regnet")
regnet= torchvision.models.regnet_y_16gf(weights="IMAGENET1K_SWAG_E2E_V1")
modules = list(regnet.children())[:-2]
regnet = nn.Sequential(*modules)
print(torchinfo.summary(regnet, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
