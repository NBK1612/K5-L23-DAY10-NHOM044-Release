"""Kiểm tra phần N1. Chỉ đọc delivery, không sửa file đầu vào."""

import hashlib
import json
from pathlib import Path


def check_detection(delivery):
    delivery = Path(delivery).resolve()
    manifest = json.loads(
        (delivery / "manifest.json").read_text(encoding="utf-8")
    )
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

    return errors
