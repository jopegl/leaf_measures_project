import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from aspp import build_aspp
from decoder import build_decoder
from backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

        self.area_head = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size=1),
            nn.Upsample(mode = 'bilinear', align_corners=True, scale_factor=4)
        )

        self.low_levet_changes = nn.Sequential (
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.measures_head = nn.Sequential (
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128,  kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64,  kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(64, 3),
            nn.Sigmoid(),
            
        )

        self.changes_after_aspp = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(mode = 'bilinear', align_corners=True, scale_factor=4)
        )

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x_aspp = x
        seg_out_raw = self.decoder(x_aspp, low_level_feat)
        low_level_feat = self.low_levet_changes(low_level_feat)
        x_aspp = self.changes_after_aspp(x_aspp)

        seg_out = F.interpolate(seg_out_raw, size=input.size()[2:], mode='bilinear', align_corners=True)
        concated = torch.concat([x_aspp, low_level_feat], dim = 1)

        area_out = self.area_head(concated)
        measures_out = self.measures_head(concated)

        return seg_out, low_level_feat, area_out, measures_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder, self.area_head, self.low_levet_changes, self.changes_after_aspp, self.measures_head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


