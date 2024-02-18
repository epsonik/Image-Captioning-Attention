import torchinfo
import torchvision
resnet152= torchvision.models.resnet152(weights="IMAGENET1K_V2")
print(torchinfo.summary(resnet152, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
densenet161 = torchvision.models.densenet161(pretrained=True)
print(torchinfo.summary(densenet161, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))


densenet121 = torchvision.models.densenet121(pretrained=True)
print(torchinfo.summary(densenet121, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0))
