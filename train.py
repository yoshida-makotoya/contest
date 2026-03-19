import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm


# -----------------------------
#  花検出（簡易）→ crop を返す
# -----------------------------
@dataclass
class CropConfig:
    max_crops: int = 1          # 1つだけ切り出す（複数切るなら増やす）
    min_area_ratio: float = 0.01  # 画像面積に対して小さすぎる領域は無視
    margin: float = 0.15        # bboxを少し広げる
    use_center_fallback: bool = True


class FlowerCropper:
    """
    OpenCVで「花っぽい色（白〜淡いピンク）」を拾って輪郭からbboxを作る簡易検出。
    条件が悪いと外すので、外したらセンタークロップへフォールバック。
    """

    def __init__(self, cfg: CropConfig = CropConfig()):
        self.cfg = cfg

    def _clip_box(self, x1, y1, x2, y2, w, h):
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return None
        return x1, y1, x2, y2

    def detect_boxes(self, bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        img_area = h * w

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ---- 白っぽい花（低彩度 & 高明度） ----
        # Sが低く、Vが高い領域
        white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))

        # ---- ピンク系（Hが赤〜紫寄り、S中、V中以上） ----
        # OpenCVのHは0-179。ピンクは大体 160-179 と 0-15 あたりに出やすい
        pink1 = cv2.inRange(hsv, (0, 40, 120), (15, 255, 255))
        pink2 = cv2.inRange(hsv, (160, 40, 120), (179, 255, 255))
        pink_mask = cv2.bitwise_or(pink1, pink2)

        mask = cv2.bitwise_or(white_mask, pink_mask)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 輪郭抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < self.cfg.min_area_ratio * img_area:
                continue
            boxes.append((x, y, x + bw, y + bh))

        # 大きい順に採用
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[: self.cfg.max_crops]

    def crop_pil(self, pil_img: Image.Image) -> List[Image.Image]:
        # PIL -> BGR
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        boxes = self.detect_boxes(bgr)
        crops: List[Image.Image] = []

        if len(boxes) == 0 and self.cfg.use_center_fallback:
            # センタークロップ（正方形）
            side = min(w, h)
            x1 = (w - side) // 2
            y1 = (h - side) // 2
            crops.append(pil_img.crop((x1, y1, x1 + side, y1 + side)))
            return crops

        for (x1, y1, x2, y2) in boxes:
            bw = x2 - x1
            bh = y2 - y1
            mx = int(bw * self.cfg.margin)
            my = int(bh * self.cfg.margin)
            box = self._clip_box(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)
            if box is None:
                continue
            cx1, cy1, cx2, cy2 = box
            crops.append(pil_img.crop((cx1, cy1, cx2, cy2)))

        if len(crops) == 0 and self.cfg.use_center_fallback:
            side = min(w, h)
            x1 = (w - side) // 2
            y1 = (h - side) // 2
            crops.append(pil_img.crop((x1, y1, x1 + side, y1 + side)))

        return crops


# -----------------------------
# Dataset（読み込み時に crop → augment）
# -----------------------------
class UmeSakuraDataset(Dataset):
    def __init__(self, root: str, class_to_idx: dict, train: bool, cropper: FlowerCropper, image_size: int = 224):
        self.root = root
        self.class_to_idx = class_to_idx
        self.train = train
        self.cropper = cropper
        self.image_size = image_size

        self.samples = []
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            if cls not in class_to_idx:
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    self.samples.append((os.path.join(cls_dir, fn), class_to_idx[cls]))

        if self.train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        crops = self.cropper.crop_pil(img)
        # 学習時はランダムに1枚、評価時は1枚目
        crop = random.choice(crops) if self.train else crops[0]

        x = self.tf(crop)
        return x, label


# -----------------------------
#  学習ユーティリティ
# -----------------------------
def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optim, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), y)

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


# -----------------------------
#  main
# -----------------------------
def main():
    # ---- 設定 ----
    data_dir = "dataset"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    class_to_idx = {"ume": 0, "sakura": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image_size = 224
    batch_size = 8
    epochs = 15
    lr = 3e-4
    weight_decay = 1e-3

    # FlowerCropper設定（必要ならここを調整）
    cropper = FlowerCropper(CropConfig(
        max_crops=1,
        min_area_ratio=0.01,
        margin=0.15,
        use_center_fallback=True,
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- Dataset ----
    train_ds = UmeSakuraDataset(train_dir, class_to_idx, train=True, cropper=cropper, image_size=image_size)
    val_ds = UmeSakuraDataset(val_dir, class_to_idx, train=False, cropper=cropper, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---- モデル（転移学習） ----
    # 軽量なら "mobilenetv3_large_100"、精度寄りなら "efficientnet_b0" などもおすすめ
    #backbone = "resnet18"
    backbone = "mobilenetv3_large_100"
    model = timm.create_model(backbone, pretrained=True, num_classes=2)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/best_ume_sakura.pth"

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, scaler, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, device)

        print(f"[{ep:02d}/{epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "backbone": backbone,
                "class_to_idx": class_to_idx,
                "image_size": image_size,
            }, best_path)
            print(f"  -> saved best: {best_path} (val_acc={best_val_acc:.4f})")

    print("done. best_val_acc:", best_val_acc)
    print("classes:", idx_to_class)


if __name__ == "__main__":
    main()