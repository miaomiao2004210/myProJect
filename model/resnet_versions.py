# model/resnet_versions.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ----------------------------
# SE 模块定义（仅 v2.0 使用）
# ----------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ----------------------------
# ResNet18-v1.0：标准版（无 SE）
# ----------------------------
class ResNet18_v1(nn.Module):
    def __init__(self, num_classes=38):
        super(ResNet18_v1, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=None)
        # 冻结前几层可选（此处不冻结，全微调）
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # 包含 avgpool
        self.classifier = nn.Linear(512, num_classes)  # 原始全连接层

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ----------------------------
# ResNet18-v2.0：带 SE + 分类头优化
# ----------------------------
class ResNet18_v2(nn.Module):
    def __init__(self, num_classes=38):
        super(ResNet18_v2, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=None)

        # 提取原始层
        layers = list(backbone.children())
        self.layer0 = nn.Sequential(*layers[:4])  # conv1 + bn1 + relu + maxpool
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]
        self.avgpool = layers[8]

        # 在每个残差块后加 SE（共4个stage，每个stage末尾加1个SE）
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # 优化分类头：可加 Dropout 或直接保持简洁（你提到“去除冗余神经元”）
        # 这里我们保持线性层，但确保无冗余（512 → 38 已是最简）
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ----------------------------
# ResNet18SE：与 ResNet18_v2 功能一致，用于 train.py 兼容
# ----------------------------
class ResNet18SE(nn.Module):
    def __init__(self, num_classes=38):
        super(ResNet18SE, self).__init__()
        backbone = models.resnet18(weights=None)  # 从头训练

        layers = list(backbone.children())
        self.layer0 = nn.Sequential(*layers[:4])  # conv1 + bn1 + relu + maxpool
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]
        self.avgpool = layers[8]

        # 添加 SE 模块
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        self.classifier = nn.Linear(512, num_classes)

        # 为了兼容 train.py 中的 model.features[0] 访问，
        # 我们构建一个 .features 属性，包含从输入到 avgpool 的所有层
        self.features = nn.Sequential(
            self.layer0,
            self.layer1,
            self.se1,
            self.layer2,
            self.se2,
            self.layer3,
            self.se3,
            self.layer4,
            self.se4,
            self.avgpool
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = self.layer4(x)
        x = self.se4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x