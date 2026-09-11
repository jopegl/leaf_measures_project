import torch.nn as nn
import torchvision.models as models


class ResNetRegressor(nn.Module):
    """Backbone ResNet pré-treinada (ImageNet) + head MLP de regressão."""

    def __init__(self, backbone_name="resnet50", num_outputs=4, pretrained=True):
        super().__init__()

        backbone_fn = getattr(models, backbone_name)

        if pretrained:
            weights_enum = getattr(models, f"{self._weights_name(backbone_name)}_Weights")
            backbone = backbone_fn(weights=weights_enum.DEFAULT)
        else:
            backbone = backbone_fn(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )

    @staticmethod
    def _weights_name(backbone_name: str) -> str:
        return backbone_name.replace("resnet", "ResNet")

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def backbone_params(self):
        return self.backbone.parameters()

    def head_params(self):
        return self.head.parameters()

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable
