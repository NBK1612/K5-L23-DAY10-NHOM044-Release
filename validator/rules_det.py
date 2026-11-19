"""Kiểm tra phần N1. Chỉ đọc delivery, không sửa file đầu vào."""

import hashlib
import json
import math
from pathlib import Path


# Cùng dung sai mép ảnh với bộ kiểm bàn giao của đề.
OVERFLOW_TOL = 0.01


def _check_label(delivery, item, valid_classes):
    label_path = (delivery / item["label"]).resolve()
    if not label_path.is_relative_to(delivery):
        raise ValueError("Đường dẫn nhãn phải nằm trong delivery")

    invalid_class = False
    overflow = False
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

    # Xét toàn file để ưu tiên ID sai dù nó nằm ở dòng sau.
    if invalid_class:
        return "invalid_class"
    if overflow:
        return "bbox_out_of_bounds"
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

        code = _check_label(delivery, item, valid_classes)
        if code is not None:
            errors.append({"type": code, "target": image_name})

    return errors
