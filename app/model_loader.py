# app/model_loader.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Encoder ────────────────────────────────────────────────────────────────

def _make_1ch_resnet34(pretrained: bool = True):
    """ResNet‑34 с одноканальным входом и stride=2"""
    from torchvision.models import resnet34, ResNet34_Weights
    m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    w = m.conv1.weight.data.mean(1, keepdim=True)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight.data = w
    return m


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = _make_1ch_resnet34(pretrained)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.l1(f0)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return f0, f1, f2, f3, f4


# ── Decoder blocks ─────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], 1)
        return self.conv(x)


class DecoderBlockNoUp(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = torch.cat([x, skip], 1)
        return self.conv(x)


# ── Full network ───────────────────────────────────────────────────────────

class MammoNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_birads: int = 6, pretrained: bool = True):
        super().__init__()
        self.enc = Encoder(pretrained)

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlockNoUp(64, 64, 32)

        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.30)
        self.cls_fc = nn.Linear(512, n_classes)
        self.br_fc = nn.Linear(512, n_birads - 1)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.enc(x)

        d = self.dec4(f4, f3)
        d = self.dec3(d, f2)
        d = self.dec2(d, f1)
        d = self.dec1(d, f0)

        seg = torch.sigmoid(self.seg_head(d))
        seg = F.interpolate(seg, scale_factor=2, mode="bilinear", align_corners=False)

        g = self.gpool(f4).flatten(1)
        logits_cls = self.cls_fc(self.drop(g))
        logits_br = self.br_fc(self.drop(g))

        return seg, logits_cls, logits_br


# ── Loader function ──────────────────────────────────────────────────────────

def load_model(model_path):
    model = MammoNet(n_classes=4, n_birads=6, pretrained=False)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model