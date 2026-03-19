import sys
import torch
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CropConfig:
    max_crops: int = 1
    min_area_ratio: float = 0.01
    margin: float = 0.15
    use_center_fallback: bool = True


class FlowerCropper:
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
        white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255))
        pink1 = cv2.inRange(hsv, (0, 40, 120), (15, 255, 255))
        pink2 = cv2.inRange(hsv, (160, 40, 120), (179, 255, 255))
        pink_mask = cv2.bitwise_or(pink1, pink2)
        mask = cv2.bitwise_or(white_mask, pink_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < self.cfg.min_area_ratio * img_area:
                continue
            boxes.append((x, y, x + bw, y + bh))

        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return boxes[: self.cfg.max_crops]

    def crop_pil(self, pil_img: Image.Image) -> List[Image.Image]:
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        boxes = self.detect_boxes(bgr)
        crops = []

        if len(boxes) == 0 and self.cfg.use_center_fallback:
            side = min(w, h)
            x1 = (w - side) // 2
            y1 = (h - side) // 2
            return [pil_img.crop((x1, y1, x1 + side, y1 + side))]

        for (x1, y1, x2, y2) in boxes:
            bw, bh = (x2 - x1), (y2 - y1)
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


def main():
    if len(sys.argv) < 3:
        print("usage: python infer_ume_sakura.py checkpoints/best_ume_sakura.pth path/to/image.jpg")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    img_path = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt["backbone"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    image_size = ckpt["image_size"]

    model = timm.create_model(backbone, pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    cropper = FlowerCropper(CropConfig())

    img = Image.open(img_path).convert("RGB")
    crops = cropper.crop_pil(img)

    # 1枚だけで判定（複数cropなら平均するのもアリ）
    x = tf(crops[0]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred = int(prob.argmax())
    print("pred:", idx_to_class[pred])
    print("prob ume:", float(prob[class_to_idx["ume"]]))
    print("prob sakura:", float(prob[class_to_idx["sakura"]]))


if __name__ == "__main__":
    main()