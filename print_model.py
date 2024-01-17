import torchinfo
import torchvision
resnet = torchvision.models.torchvision.models.regnet_y_32gf(weights="IMAGENET1K_SWAG_E2E_V1")
torchinfo.summary(resnet, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0)
