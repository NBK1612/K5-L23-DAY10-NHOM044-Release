"""Lỗi luật metadata + panoptic (phần của N2).

Bốn loại lỗi:
  - scene_leak          : một scene_id có ảnh ở cả train và val. target = scene_id.
  - guideline_mismatch  : guideline_version của ảnh khác guideline của job (jobs.csv).
  - invalid_category_id : category_id của segment không có trong panoptic.json/categories.
  - thing_overlap       : area khai trong segments_info lớn hơn số pixel thật của segment trong mask.

Chỉ đọc thư mục delivery được truyền vào; chỉ thư viện chuẩn + scripts/pngutil.py.
Không giả định tên ảnh dạng COCO: mọi liên kết đi qua manifest.json.

Mỗi hàm trả về list[{"type": ..., "target": ...}].
Chạy thử riêng: python validator/rules_meta.py <delivery> [--out meta.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pngutil  # noqa: E402  (đọc PNG bằng thư viện chuẩn)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(delivery: Path) -> list[dict]:
    return _load_json(delivery / "metadata.json")["items"]


def _jobs(delivery: Path) -> dict[str, dict]:
    with (delivery / "jobs.csv").open(encoding="utf-8", newline="") as f:
        return {row["job_id"].strip(): row for row in csv.DictReader(f)}


# ---------------------------------------------------------------- metadata

def check_scene_leak(delivery: Path) -> list[dict]:
    """Cảnh có ảnh ở hơn một split -> một lỗi, target là scene_id."""
    splits: dict[str, set[str]] = {}
    for item in _metadata(delivery):
        splits.setdefault(item["scene_id"], set()).add(item["split"].strip())
    return [{"type": "scene_leak", "target": scene}
            for scene, seen in sorted(splits.items()) if len(seen) > 1]


def check_guideline_mismatch(delivery: Path) -> list[dict]:
    """So guideline của ảnh với guideline ghi ở job của nó (không so với ảnh khác)."""
    jobs = _jobs(delivery)
    out = []
    for item in _metadata(delivery):
        job = jobs.get(item["job_id"].strip())
        if job is None:
            continue  # job không tồn tại: không thuộc 4 loại lỗi này
        if item["guideline_version"].strip() != job["guideline"].strip():
            out.append({"type": "guideline_mismatch", "target": item["image"]})
    return out


# ---------------------------------------------------------------- panoptic

def _panoptic_by_image(delivery: Path) -> tuple[dict, dict[str, dict]]:
    """Trả về (panoptic, {tên ảnh: annotation}).

    Ghép annotation với ảnh qua tên file mask ghi trong manifest.json,
    nên vẫn đúng khi đề ẩn đổi tên ảnh / đánh số lại id.
    """
    pan = _load_json(delivery / "panoptic.json")
    ann_by_file = {a["file_name"]: a for a in pan["annotations"]}
    manifest = _load_json(delivery / "manifest.json")["items"]
    out = {}
    for name, item in manifest.items():
        ann = ann_by_file.get(Path(item["mask"]).name)
        if ann is not None:
            out[name] = ann
    return pan, out


def check_invalid_category_id(delivery: Path) -> list[dict]:
    pan, anns = _panoptic_by_image(delivery)
    valid = {c["id"] for c in pan["categories"]}
    return [{"type": "invalid_category_id", "target": name}
            for name, ann in anns.items()
            if any(s["category_id"] not in valid for s in ann["segments_info"])]


def _pixel_counts(mask_path: Path) -> Counter:
    """Đếm pixel theo segment id = R + G*256 + B*256^2."""
    _, _, px = pngutil.read_rgb(mask_path)
    counts = Counter(zip(px[0::3], px[1::3], px[2::3]))  # đếm theo bộ (R, G, B), chạy bằng C nên nhanh
    return Counter({r + g * 256 + b * 65536: n for (r, g, b), n in counts.items()})


def check_thing_overlap(delivery: Path) -> list[dict]:
    """area khai > số pixel thật -> segment đã bị vật khác đè (pixel bị lấy mất)."""
    _, anns = _panoptic_by_image(delivery)
    manifest = _load_json(delivery / "manifest.json")["items"]
    out = []
    for name, ann in anns.items():
        mask_path = delivery / manifest[name]["mask"]
        if not mask_path.is_file():
            continue  # thiếu mask: không đếm được, không đoán
        counts = _pixel_counts(mask_path)
        for seg in ann["segments_info"]:
            if seg["id"] == 0:
                continue  # id 0 là nền
            if seg["area"] > counts.get(seg["id"], 0):
                out.append({"type": "thing_overlap", "target": name})
                break
    return out


# ---------------------------------------------------------------- gộp

CHECKS = [check_scene_leak, check_guideline_mismatch, check_invalid_category_id, check_thing_overlap]


def check_meta(delivery) -> list[dict]:
    """Chạy cả 4 hàm. Việc chọn một lỗi/ảnh theo thứ tự ưu tiên do validate.py lo."""
    delivery = Path(delivery)
    out = []
    for fn in CHECKS:
        out.extend(fn(delivery))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Lỗi luật metadata + panoptic")
    ap.add_argument("delivery")
    ap.add_argument("--out")
    args = ap.parse_args()
    errors = check_meta(args.delivery)
    for kind, n in sorted(Counter(e["type"] for e in errors).items()):
        print(f"{kind}: {n}")
    print(f"tổng: {len(errors)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"errors": errors}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
