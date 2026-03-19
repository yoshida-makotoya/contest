import os
import re
import time
import random
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image

# =========================
# 設定
# =========================
ROOT_DIR = "dataset"
TRAIN_RATIO = 0.8
PER_CLASS_LIMIT = 300          # 各カテゴリから最大何枚保存するか
MIN_WIDTH = 224                # 小さすぎる画像を除外
MIN_HEIGHT = 224
REQUEST_TIMEOUT = 30
SLEEP_SEC = 0.2                # API連打しすぎ防止
USER_AGENT = "ume-sakura-dataset-downloader/1.0"

# Wikimedia Commons のカテゴリ
TARGETS = {
    "ume": [
        "Category:Prunus mume",
    ],
    "sakura": [
        "Category:Cherry blossoms",
        "Category:Prunus serrulata",
    ],
}

API_URL = "https://commons.wikimedia.org/w/api.php"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def ensure_dirs():
    for split in ["train", "valid"]:
        for cls in TARGETS.keys():
            os.makedirs(os.path.join(ROOT_DIR, split, cls), exist_ok=True)


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


def get_extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return ext
    return ".jpg"


def is_allowed_image_title(title: str) -> bool:
    title_lower = title.lower()
    ng_ext = [".svg", ".gif", ".tif", ".tiff", ".pdf", ".djvu"]
    return not any(title_lower.endswith(ext) for ext in ng_ext)


def fetch_category_members(category_title: str, cmtype="file", cmlimit=500):
    """
    Wikimedia Commons のカテゴリ内メンバーを列挙
    """
    members = []
    cont = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": cmtype,
            "cmlimit": cmlimit,
        }
        if cont:
            params["cmcontinue"] = cont

        r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        batch = data.get("query", {}).get("categorymembers", [])
        members.extend(batch)

        if "continue" not in data:
            break

        cont = data["continue"]["cmcontinue"]
        time.sleep(SLEEP_SEC)

    return members


def fetch_image_info(titles):
    """
    ファイルタイトルの一覧から画像URL等をまとめて取得
    titles: ["File:xxx.jpg", "File:yyy.png", ...]
    """
    if not titles:
        return []

    results = []
    chunk_size = 50

    for i in range(0, len(titles), chunk_size):
        chunk = titles[i:i + chunk_size]
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "titles": "|".join(chunk),
        }

        r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            title = page.get("title")
            infos = page.get("imageinfo", [])
            if not infos:
                continue

            info = infos[0]
            results.append({
                "title": title,
                "url": info.get("url"),
                "width": info.get("width"),
                "height": info.get("height"),
                "mime": info.get("mime"),
            })

        time.sleep(SLEEP_SEC)

    return results


def download_and_validate_image(url: str):
    """
    画像を取得して Pillow で開けるか確認
    戻り値: PIL.Image or None
    """
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content))
        img.load()

        if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
            return None

        return img

    except Exception:
        return None


def save_as_jpeg(img: Image.Image, save_path: str):
    """
    形式を問わず JPEG として保存
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(save_path, format="JPEG", quality=95)


def choose_split(train_ratio=0.8):
    return "train" if random.random() < train_ratio else "valid"


def collect_titles_for_class(class_name: str):
    """
    クラスに属する複数カテゴリから File タイトルを集める
    """
    all_titles = []

    for category in TARGETS[class_name]:
        print(f"[INFO] カテゴリ取得中: {category}")
        members = fetch_category_members(category, cmtype="file")
        for m in members:
            title = m.get("title", "")
            if title.startswith("File:") and is_allowed_image_title(title):
                all_titles.append(title)

    # 重複除去
    all_titles = list(dict.fromkeys(all_titles))
    return all_titles


def collect_images_for_class(class_name: str, limit: int):
    print(f"\n===== {class_name} の収集中 =====")

    titles = collect_titles_for_class(class_name)
    print(f"[INFO] タイトル候補数: {len(titles)}")

    infos = fetch_image_info(titles)
    print(f"[INFO] 画像情報取得数: {len(infos)}")

    saved = 0
    skipped = 0

    for idx, info in enumerate(infos, start=1):
        if saved >= limit:
            break

        url = info.get("url")
        title = info.get("title", f"{class_name}_{idx}")
        width = info.get("width") or 0
        height = info.get("height") or 0

        if not url:
            skipped += 1
            continue

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            skipped += 1
            continue

        img = download_and_validate_image(url)
        if img is None:
            skipped += 1
            continue

        split = choose_split(TRAIN_RATIO)
        out_dir = os.path.join(ROOT_DIR, split, class_name)

        base_name = sanitize_filename(os.path.splitext(title.replace("File:", ""))[0])
        save_name = f"{class_name}_{saved+1:04d}_{base_name}.jpg"
        save_path = os.path.join(out_dir, save_name)

        try:
            save_as_jpeg(img, save_path)
            saved += 1
            print(f"[SAVE] {save_path}")
        except Exception:
            skipped += 1

        time.sleep(SLEEP_SEC)

    print(f"[DONE] {class_name}: saved={saved}, skipped={skipped}")


def main():
    random.seed(42)
    ensure_dirs()

    for class_name in TARGETS.keys():
        collect_images_for_class(class_name, PER_CLASS_LIMIT)

    print("\n完了しました。")
    print(f"保存先: {os.path.abspath(ROOT_DIR)}")


if __name__ == "__main__":
    main()