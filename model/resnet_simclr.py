import torch.nn as nn
import torchvision.models as models
from exception.exception import InvalidBackboneError
import torch.nn.functional as F
# class ResNetSimCLR(nn.Module):
#     def __init__(self, base_model, out_dim):
#         super(ResNetSimCLR, self).__init__()
#         self.resnet_dict = {
#             "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
#             "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)
#         }
#         backbone = self._get_basemodel(base_model)
#         dim_mlp = backbone.fc.in_features

#         # TAG: Remove the last fc layer of the ResNet
#         self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
#         # Tag: add mlp projection head 
#         self.mlp = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), backbone.fc)

#     def forward(self, x):
#         out1 = self.backbone(x)
#         out1 = torch.flatten(out1, start_dim=1)
#         out2 = self.mlp(out1)
#         return out2

#     def _get_basemodel(self, model_name):
#         try:
#             model = self.resnet_dict[model_name]
#             # print(model)
#         except KeyError:
#             raise InvalidBackboneError(
#                 "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
#             )
#         else:
#             return model



class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
        #                     "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        # self.backbone = self._get_basemodel(base_model)
        self.backbone = ResNet50(num_classes=out_dim)
        hidden_dim = self.backbone.fc.in_features

        # Customize for CIFAR10. Replace cov 7×7 with conv 3×3, and remove the first max pooling.
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.backbone.maxpool = nn.Identity()
        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False))
            
    # def _get_basemodel(self, model_name):
    #     try:
    #         model = self.resnet_dict[model_name]
    #     except KeyError:
    #         raise InvalidBackboneError(
    #             "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
    #     else:
    #         return model

    def forward(self, x):
        return self.backbone(x)










"""ResNet 50 for CIFAR10"""
# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 1×1 Conv
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3×3 Conv
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1×1 Conv
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  #flatten
        out = self.fc(out)
        return out


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)