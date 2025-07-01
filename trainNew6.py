# train.py – версия без аугментаций и с «чистым» MammoNet
"""
Запуск:  python train.py      (или  py -3.10 train.py)

Главные отличия от старой версии
─────────────────────────────────
1. Используем **datasets_no_aug.MammoNPZ** – никаких случайных трансформ.
2. Модель берём из **model_clean.MammoNet** (256²‑скип, маска 512²).
3. Исправлен расчёт κ‑метрики для ординального BI‑RADS:
   вместо 5‑логитного тензора передаём в `metric_kap` итоговый индекс класса.
4. Обновлён вызов mixed‑precision: `torch.amp.autocast(device_type="cuda")`.

GPU‑требования: 8 ГБ VRAM (RTX 4060 Laptop) выдерживает batch = 8 под FP16.
"""

import time, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# ── наши модули ────────────────────────────────────────────────────────────
from datasets import MammoNPZ
from model import MammoNet
from losses import FocalTversky, SoftF1, OrdinalCE

# ── torch & metrics ────────────────────────────────────────────────────────
from torch.optim import AdamW
from torch.amp import autocast                # новый путь c PyTorch 2.2+
from torch.amp import GradScaler              # GradScaler живёт здесь же
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassCohenKappa,
)
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Собственная реализация FocalLoss (без внешних зависимостей)

try:
    from focalloss import FocalLoss
except ImportError:
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=None, gamma=1.2):
            super().__init__(); self.alpha, self.gamma = alpha, gamma
            self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        def forward(self, logits, target):
            ce = self.ce(logits, target); pt = torch.exp(-ce)
            loss = ((1-pt)**self.gamma) * ce
            if self.alpha is not None:
                loss = self.alpha[target] * loss
            return loss.mean()

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa

SEED, FOLD = 2025, 0
NPZ_PATH = Path(r"D:\Mag\sem2\standarts\Datase\npz\mammo_dataset4.npz")
BATCH = 8; EPOCHS = 150
LR_BACKBONE = 1e-5; LR_HEADS = 5e-5; PATIENCE_ES = 15

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = np.load(NPZ_PATH, mmap_mode="r")["C"]
train_idx, val_idx = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                          .split(np.zeros(len(labels)), labels))[FOLD]
train_ds, val_ds = MammoNPZ(NPZ_PATH, train_idx, True), MammoNPZ(NPZ_PATH, val_idx, False)

cls_weights = torch.tensor([2.0, 1.5, 2.0, 0.2], device=device)
weights_np  = cls_weights.cpu().numpy()[train_ds.C]
train_ld = DataLoader(train_ds, BATCH,
                      sampler=WeightedRandomSampler(weights_np, len(train_ds), replacement=True),
                      num_workers=0, pin_memory=True)
val_ld   = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0, pin_memory=True)

net = MammoNet(pretrained=True).to(device)
# ── резюме: дообучаться с последнего чекпойнта ───────────────
CKPT_PATH = Path(f"best_fold{FOLD}.pt")
if CKPT_PATH.exists():
    net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"Loaded checkpoint {CKPT_PATH} — continuing training")
params = [
    {"params": net.enc.parameters(), "lr": LR_BACKBONE},
    {"params": (
        list(net.dec4.parameters()) + list(net.dec3.parameters()) +
        list(net.dec2.parameters()) + list(net.dec1.parameters()) +
        [net.seg_head.weight, net.seg_head.bias,
         net.cls_fc.weight, net.cls_fc.bias,
         net.br_fc.weight,  net.br_fc.bias]
    ), "lr": LR_HEADS},
]
opt    = torch.optim.AdamW(params, weight_decay=5e-4)
scaler = GradScaler()

loss_seg = FocalTversky()
loss_cls = FocalLoss(alpha=cls_weights, gamma=1.2)
loss_br  = OrdinalCE()

metric_f1  = MulticlassF1Score(num_classes=4, average="macro").to(device)
metric_kap = MulticlassCohenKappa(num_classes=6, weights="quadratic").to(device)

scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

best_score, es_cnt = -1.0, 0
for epoch in range(1, EPOCHS + 1):
    net.train(); ep_loss, t0 = 0.0, time.time()
    for img, msk, cls, bir in tqdm(train_ld, desc=f"Epoch {epoch}"):
        img, msk = img.to(device), msk.to(device); cls, bir = cls.to(device), bir.to(device)
        bir_idx  = (bir.long() - 1).clamp(0, 5)
        with autocast():
            seg, lg_cls, lg_br = net(img)
            loss = loss_seg(seg, msk) + loss_cls(lg_cls, cls) + 1.8 * loss_br(lg_br, bir_idx)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(True)
        ep_loss += loss.item()

    net.eval(); metric_f1.reset(); metric_kap.reset()
    with torch.no_grad():
        for img, _, cls, bir in val_ld:
            img, cls = img.to(device), cls.to(device)
            bir_idx  = (bir.long() - 1).clamp(0, 5).to(device)
            _, lg_cls, lg_br = net(img)
            metric_f1.update(lg_cls, cls)
            metric_kap.update((lg_br.sigmoid() > 0.5).sum(1), bir_idx)
    f1, kappa = metric_f1.compute().item(), metric_kap.compute().item();
    score = f1 + (kappa if not torch.isnan(torch.tensor(kappa)) else 0)
    print(f"[{epoch:03d}] loss={ep_loss/len(train_ld):.3f}  F1={f1:.3f}  κ={kappa:.3f}  lr={opt.param_groups[1]['lr']:.1e}")
    scheduler.step(f1)
    if score > best_score + 1e-4:
        best_score = score; torch.save(net.state_dict(), f"best_fold{FOLD}.pt"); es_cnt = 0; print("  ↳ new best", best_score)
    else:
        es_cnt += 1
        if es_cnt >= PATIENCE_ES:
            print(f"Early‑Stopping after {epoch} epochs"); break

    """
best_fold05.pt

Test samples: 960 (fold 0)
Thresholds: [0.5 0.3 0.5 0.3]

==== 4-class detailed report (with thresholds) ====
CALC    : AUROC=0.928  Prec=0.801  Rec=0.671  Spec=0.944  F1=0.730
CIRC    : AUROC=0.963  Prec=0.789  Rec=0.917  Spec=0.918  F1=0.848
MISC    : AUROC=0.891  Prec=0.665  Rec=0.688  Spec=0.885  F1=0.676
NORM    : AUROC=0.999  Prec=0.983  Rec=0.950  Spec=0.994  F1=0.966
Overall accuracy: 0.80625

Confusion-matrix (rows = preds, cols = true):
 [[161  22  57   0]
 [  4 220  16   0]
 [ 36  35 165   4]
 [  0   2  10 228]]

==== BI-RADS ord ====
Quadratic κ: 0.8093454837799072
    """

