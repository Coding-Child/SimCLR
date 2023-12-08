import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )

        print(verbose)

    return model


class ResNetSimCLR(nn.Module):
    def __init__(self, hidden_size, projection_dim):
        super(ResNetSimCLR, self).__init__()

        self.model = resnet50(pretrained=True, key='MoCoV2', progress=False)
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(start_dim=1),
                                        nn.Linear(2048, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, projection_dim))

    def forward(self, x):
        feature = self.model(x)
        out = self.projection(feature)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = ResNetSimCLR(hidden_size=4096, projection_dim=128).to(device)
    del model.projection
    print(model)

