import torchvision
import torchinfo
from torch import nn
print("inception")
inception = torchvision.models.inception_v3(pretrained=True)
print(torchinfo.summary(inception, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
#
print("resnet152")
resnet152 = torchvision.models.resnet152(weights="IMAGENET1K_V2")
print(torchinfo.summary(resnet152, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("resnet101")
resnet101 = torchvision.models.resnet101(pretrained=True)
print(torchinfo.summary(resnet101, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))

print("dense161")
densenet161 = torchvision.models.densenet161(pretrained=True)
print(torchinfo.summary(densenet161, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("dense121")
densenet121 = torchvision.models.densenet121(pretrained=True)
print(torchinfo.summary(densenet121, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("dense201")
densenet201 = torchvision.models.densenet201(pretrained=True)
print(torchinfo.summary(densenet201, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
print("regnet")
densenet201 = torchvision.models.regnet_y_16gf(weights="IMAGENET1K_SWAG_E2E_V1")
print(torchinfo.summary(densenet201, (3, 224, 224), batch_dim=0,
                        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
