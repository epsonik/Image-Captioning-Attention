import torchinfo
import torchvision
regnet32 = torchvision.models.regnet_y_32gf(weights="IMAGENET1K_SWAG_E2E_V1")
print(torchinfo.summary(regnet32, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
resnet = torchvision.models.resnet101(pretrained = True)
print(torchinfo.summary(resnet, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))


regnet16 = torchvision.models.regnet_y_16gf(weights="IMAGENET1K_SWAG_E2E_V1")
print(torchinfo.summary(regnet16, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
