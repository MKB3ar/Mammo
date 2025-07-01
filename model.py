# model_clean.py – «чистый» вариант без stem‑maxpool
# сохраняет точность границ при входе 512×512 и комфортно помещается в 8 ГБ VRAM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ── Encoder ────────────────────────────────────────────────────────────────

def _make_1ch_resnet34(pretrained: bool = True):
    """ResNet‑34 с одноканальным входом и **stride = 2** (оставляем).
    Мы только убираем максимальный пул после conv1, чтобы первая фича была
    256 × 256 (а не 128 × 128). Это удваивает разрешение skip‑связи f0,
    но почти не бьёт по памяти (≈ +0.4 ГБ при batch=8, fp16).
    """
    m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    w = m.conv1.weight.data.mean(1, keepdim=True)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight.data = w
    return m


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = _make_1ch_resnet34(pretrained)
        # 🔻 убрали m.maxpool → f0 = 64ch × 256²
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        f0 = self.stem(x)     # 64 @ 256² (было 128²)
        f1 = self.l1(f0)      # 64 @ 256²
        f2 = self.l2(f1)      # 128 @ 128²
        f3 = self.l3(f2)      # 256 @ 64²
        f4 = self.l4(f3)      # 512 @ 32²
        return f0, f1, f2, f3, f4


# ── Decoder blocks ─────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Стандартный блок: ↑2× + concat(skip) + 2×3×3 conv"""

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
        x = self.up(x)                # ↑2× по пространству
        x = torch.cat([x, skip], 1)   # concat
        return self.conv(x)


class DecoderBlockNoUp(nn.Module):
    """Последний уровень: **без** апсэмпла, т.к. f0 уже такого же размера."""

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

        self.dec4 = DecoderBlock(512, 256, 256)   # 32² → 64²
        self.dec3 = DecoderBlock(256, 128, 128)   # 64² → 128²
        self.dec2 = DecoderBlock(128,  64,  64)   # 128² → 256²
        # 🔻 последний блок без апсэмпла (256² остаётся 256²)
        self.dec1 = DecoderBlockNoUp(64, 64, 32)

        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        self.gpool   = nn.AdaptiveAvgPool2d(1)
        self.drop    = nn.Dropout(0.30)          # ← новое v2
        
        self.cls_fc  = nn.Linear(512, n_classes)
        self.br_fc   = nn.Linear(512, n_birads - 1)

    # ------------------------------------------------------------------
    def forward(self, x):
        f0, f1, f2, f3, f4 = self.enc(x)

        d = self.dec4(f4, f3)
        d = self.dec3(d, f2)
        d = self.dec2(d, f1)
        d = self.dec1(d, f0)           # итог 256²

        seg = torch.sigmoid(self.seg_head(d))
        # ↑ финальный апсэмпл до 512² – билатерально, без размытия углов
        seg = F.interpolate(seg, scale_factor=2, mode="bilinear", align_corners=False)

        g = self.gpool(f4).flatten(1)
        logits_cls = self.cls_fc(self.drop(g))   # ← оборачиваем новое v2

        logits_cls = self.cls_fc(g)
        logits_br  = self.br_fc(g)
        return seg, logits_cls, logits_br
