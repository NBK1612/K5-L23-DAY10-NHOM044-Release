"""Kiểm tra phần N1. Chỉ đọc delivery, không sửa file đầu vào."""

import hashlib
import json
import math
from pathlib import Path


# Cùng dung sai mép ảnh với bộ kiểm bàn giao của đề.
OVERFLOW_TOL = 0.01
# Dung sai kích thước do nhãn YOLO được làm tròn 6 chữ số.
SIZE_TOL = 0.000002
CENTER_TOL_PX = 2.01


def _check_label(delivery, item, valid_classes, image_size):
    label_path = (delivery / item["label"]).resolve()
    if not label_path.is_relative_to(delivery):
        raise ValueError("Đường dẫn nhãn phải nằm trong delivery")

    invalid_class = False
    overflow = False
    duplicate = False
    previous_boxes = []
    image_width, image_height = image_size
    for line_no, text in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not text.strip():
            continue
        parts = text.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path.name}:{line_no}: cần 5 cột YOLO")

        class_id, cx, cy, w, h = map(float, parts)
        if (
            not math.isfinite(class_id)
            or not class_id.is_integer()
            or int(class_id) not in valid_classes
        ):
            invalid_class = True
            continue

        if not all(math.isfinite(value) for value in (cx, cy, w, h)):
            raise ValueError(f"{label_path.name}:{line_no}: tọa độ nan/inf")

        if (
            cx - w / 2 < -OVERFLOW_TOL
            or cy - h / 2 < -OVERFLOW_TOL
            or cx + w / 2 > 1 + OVERFLOW_TOL
            or cy + h / 2 > 1 + OVERFLOW_TOL
        ):
            overflow = True

        # Khung lặp: cùng lớp, gần như cùng kích thước, tâm lệch <= 2 px.
        # Không dùng riêng IoU: các vật thật có thể che/chồng lên nhau.
        for old_class, old_cx, old_cy, old_w, old_h in previous_boxes:
            if (
                int(class_id) == old_class
                and abs(w - old_w) <= SIZE_TOL
                and abs(h - old_h) <= SIZE_TOL
                and abs(cx - old_cx) * image_width <= CENTER_TOL_PX
                and abs(cy - old_cy) * image_height <= CENTER_TOL_PX
            ):
                duplicate = True
                break
        previous_boxes.append((int(class_id), cx, cy, w, h))

    # Xét toàn file để ưu tiên ID sai dù nó nằm ở dòng sau.
    if invalid_class:
        return "invalid_class"
    if overflow:
        return "bbox_out_of_bounds"
    if duplicate:
        return "duplicate_box"
    return None


def check_detection(delivery):
    delivery = Path(delivery).resolve()
    manifest = json.loads(
        (delivery / "manifest.json").read_text(encoding="utf-8")
    )
    valid_classes = {
        int(line.split(maxsplit=1)[0])
        for line in (delivery / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    panoptic = json.loads(
        (delivery / "panoptic.json").read_text(encoding="utf-8")
    )
    image_sizes = {
        Path(image["file_name"]).name: (image["width"], image["height"])
        for image in panoptic["images"]
    }
    errors = []

    for image_name, item in sorted(manifest["items"].items()):
        image_path = (delivery / item["image"]).resolve()
        if not image_path.is_relative_to(delivery):
            raise ValueError("Đường dẫn ảnh phải nằm trong delivery")

        if not image_path.is_file():
            errors.append({"type": "missing_image", "target": image_name})
            continue

        # Tính hash từ byte thật, không đọc ảnh như văn bản.
        digest = hashlib.sha256()
        with image_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        if digest.hexdigest() != item["image_sha256"]:
            errors.append({"type": "wrong_hash", "target": image_name})
            continue

        image_size = image_sizes[image_path.name]
        code = _check_label(delivery, item, valid_classes, image_size)
        if code is not None:
            errors.append({"type": code, "target": image_name})

    return errors
