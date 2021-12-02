import torch.nn as nn
import torchvision.models as models
import torch
from exception.exception import InvalidBackboneError
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
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)

        hidden_dim = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)