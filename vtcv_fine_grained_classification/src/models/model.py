import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels import xception


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y)))).view(b, c, 1, 1)
        return x * y


class Xception_V2_1(nn.Module):
    """Xception up to Block 11, then SE block, global average pooling and a linear head.

    The exit flow (block12, conv3, conv4) stays in self.model so old checkpoints still load,
    but it is never run.
    """

    def __init__(self, num_classes=32, imagenet=True):
        super().__init__()
        self.model = xception(num_classes=1000, pretrained="imagenet" if imagenet else None)
        self.model.last_linear = nn.Linear(728, num_classes)
        self.se_block = SEBlock(728)
        self.block11 = self.model.block11
        self.model.block11 = nn.Identity()
        self.frozen = []
        self._init_weights(imagenet)

    def _init_weights(self, imagenet):
        if not imagenet:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for m in list(self.se_block.modules()) + [self.model.last_linear]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def blocks(self):
        stem = [self.model.conv1, self.model.bn1, self.model.conv2, self.model.bn2]
        return stem, [getattr(self.model, f"block{i}") for i in range(1, 11)] + [self.block11]

    def freeze_first_blocks(self, n_blocks):
        stem, blocks = self.blocks()
        self.frozen = stem + blocks[:n_blocks]
        for m in self.frozen:
            for p in m.parameters():
                p.requires_grad = False
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:
            m.eval()
        return self

    def forward(self, x):
        x = (x - 0.5) / 0.5
        x = F.relu(self.model.bn1(self.model.conv1(x)))
        x = F.relu(self.model.bn2(self.model.conv2(x)))
        _, blocks = self.blocks()
        for block in blocks[:-1]:
            x = block(x)
        ftm = self.block11(x)
        if torch.is_grad_enabled() and not ftm.requires_grad:
            ftm.requires_grad_()

        pooled = F.adaptive_avg_pool2d(self.se_block(ftm), 1).flatten(1)
        logits = self.model.last_linear(pooled)
        return logits, {"feature_maps": ftm, "pooled": pooled}
