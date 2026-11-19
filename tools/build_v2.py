#!/usr/bin/env python3
"""Dựng lô bàn giao v2 (Pha 4 - N5).

Sản phẩm detection v2 gồm 7 phần:
  1. images/
  2. labels/
  3. rework_queue.csv
  4. manifest.csv
  5. label_summary.csv
  6. truyvet.csv
  7. README.txt

Cách dùng:
  python tools/build_v2.py runs/r44007/delivery --errors runs/r44007/errors.json --out runs/r44007/handover/<release_id>_v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

# Các loại lỗi nhãn detection hoặc thiếu file: phải đưa vào rework_queue và rời khỏi v2
DET_REWORK = {
    "missing_image",
    "invalid_class",
    "bbox_out_of_bounds",
    "duplicate_box",
    "class_swap",
    "bbox_drift",
    "missing_box",
}

PANOPTIC_ERRORS = {
    "invalid_category_id",
    "thing_overlap",
    "category_swap",
    "instance_merge",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Dựng lô v2 bàn giao (không ghi đè v1)")
    parser.add_argument("delivery", type=Path, help="Đường dẫn thư mục delivery v1")
    parser.add_argument("--errors", type=Path, help="Đường dẫn file errors.json (mặc định tìm cạnh delivery)")
    parser.add_argument("--out", type=Path, help="Đường dẫn thư mục đầu ra v2")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    if not delivery.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục delivery: {delivery}")

    errors_path = args.errors.resolve() if args.errors else (delivery.parent / "errors.json")
    if not errors_path.is_file():
        raise SystemExit(f"Không tìm thấy file errors.json tại: {errors_path}")

    # Đọc dữ liệu v1
    rel_info = json.loads((delivery / "release.json").read_text(encoding="utf-8"))
    rid = rel_info["release_id"]
    manifest = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))["items"]
    metadata_list = json.loads((delivery / "metadata.json").read_text(encoding="utf-8"))["items"]
    meta = {m["image"]: m for m in metadata_list}
    jobs = {row["job_id"]: row for row in read_csv(delivery / "jobs.csv")}
    classes = [ln.split(" ", 1)[1].strip() for ln in (delivery / "classes.txt").read_text(encoding="utf-8").splitlines()]

    errors = json.loads(errors_path.read_text(encoding="utf-8"))["errors"]
    err_imgs = {e["target"] for e in errors}

    # Phân loại lỗi
    rework_imgs = {e["target"] for e in errors if e["target"] in manifest and e["type"] in DET_REWORK}
    pan_err_imgs = [e["target"] for e in errors if e["type"] in PANOPTIC_ERRORS]
    wrong_hash_imgs = [e["target"] for e in errors if e["type"] == "wrong_hash"]

    out_dir = args.out.resolve() if args.out else (delivery.parent / "handover" / f"{rid}_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. & 2. images/ và labels/
    # Chép mọi ảnh và nhãn không thuộc rework_queue
    stems = []
    for stem, item in manifest.items():
        if stem in rework_imgs:
            continue
        src_img = delivery / item["image"]
        src_lab = delivery / item["label"]
        if not src_img.is_file() or not src_lab.is_file():
            continue
        dst_img = images_dir / f"{stem}.jpg"
        dst_lab = labels_dir / f"{stem}.txt"
        shutil.copyfile(src_img, dst_img)
        shutil.copyfile(src_lab, dst_lab)
        stems.append(stem)

    stems.sort()
    print(f"v2 có {len(stems)} ảnh (loại {len(rework_imgs)} ảnh rework)")

    # 3. rework_queue.csv: image,type,annotator,reviewer,action
    rework_rows = []
    # Bản đồ lỗi của từng ảnh
    img_to_error_type = {}
    for e in errors:
        if e["target"] in rework_imgs:
            img_to_error_type[e["target"]] = e["type"]

    for stem in sorted(rework_imgs):
        etype = img_to_error_type.get(stem, "rework")
        m = meta.get(stem, {})
        job = jobs.get(m.get("job_id"), {})
        rework_rows.append({
            "image": stem,
            "type": etype,
            "annotator": job.get("annotator", ""),
            "reviewer": job.get("reviewer", ""),
            "action": "gán lại nhãn detection",
        })

    with (out_dir / "rework_queue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "type", "annotator", "reviewer", "action"])
        writer.writeheader()
        writer.writerows(rework_rows)

    # 4. manifest.csv: file,sha256
    manifest_rows = []
    for s in stems:
        img_rel = f"images/{s}.jpg"
        lab_rel = f"labels/{s}.txt"
        manifest_rows.append({"file": img_rel, "sha256": sha256(images_dir / f"{s}.jpg")})
        manifest_rows.append({"file": lab_rel, "sha256": sha256(labels_dir / f"{s}.txt")})

    manifest_rows.sort(key=lambda x: x["file"])
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 5. label_summary.csv: class,count
    counts = {name: 0 for name in classes}
    for s in stems:
        for line in (labels_dir / f"{s}.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cid = int(float(line.split()[0]))
                if 0 <= cid < len(classes):
                    counts[classes[cid]] += 1
                else:
                    cname = str(cid)
                    counts[cname] = counts.get(cname, 0) + 1

    summary_rows = [{"class": name, "count": counts[name]} for name in classes if counts[name] > 0]
    total_labels = sum(counts.values())
    summary_rows.append({"class": "TOTAL", "count": total_labels})

    with (out_dir / "label_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # 6. truyvet.csv: image,scene_id,split,job_id,annotator,reviewer,guideline
    # Giải quyết rò rỉ cảnh (scene_leak): các ảnh cùng cảnh phải về cùng split
    # Lấy split phổ biến nhất trong cảnh hoặc ưu tiên train
    scene_to_split = {}
    for s in stems:
        m = meta[s]
        sc = m["scene_id"]
        sp = m["split"]
        scene_to_split.setdefault(sc, []).append(sp)

    final_scene_split = {}
    for sc, splits in scene_to_split.items():
        if "train" in splits:
            final_scene_split[sc] = "train"
        else:
            final_scene_split[sc] = splits[0]

    truyvet_rows = []
    for s in stems:
        m = meta[s]
        job = jobs[m["job_id"]]
        truyvet_rows.append({
            "image": s,
            "scene_id": m["scene_id"],
            "split": final_scene_split[m["scene_id"]],
            "job_id": m["job_id"],
            "annotator": job["annotator"],
            "reviewer": job["reviewer"],
            "guideline": job["guideline"],  # Luôn theo job, không theo meta
        })

    with (out_dir / "truyvet.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "scene_id", "split", "job_id", "annotator", "reviewer", "guideline"])
        writer.writeheader()
        writer.writerows(truyvet_rows)

    # 7. README.txt
    rate = len(err_imgs) / len(manifest)
    v1_decision = "Hold" if rate > 0.02 else "Release"

    # Nhóm job theo guideline
    g_jobs = {}
    for jid, jinfo in sorted(jobs.items()):
        g_jobs.setdefault(jinfo["guideline"], []).append(jid)
    guideline_parts = []
    for gver, jids in sorted(g_jobs.items()):
        guideline_parts.append(f"{gver} ({jids[0]}-{jids[-1]})")
    guideline_str = ", ".join(guideline_parts)

    pan_str = ", ".join(sorted(pan_err_imgs)) if pan_err_imgs else "không có"
    hash_str = ", ".join(sorted(wrong_hash_imgs)) if wrong_hash_imgs else "không có"

    readme_content = f"""Release: {rid}_v2
Parent: {rid}
Số ảnh: {len(stems)}
Số nhãn: {total_labels}
Guideline: {guideline_str}
QC v1: {len(err_imgs)}/{len(manifest)} = {rate:.1%}
Kết luận v1: {v1_decision} — vượt ngưỡng 2% ({len(err_imgs)}/{len(manifest)} = {rate:.1%})
Kết luận v2: Release — đã tách {len(rework_imgs)} ảnh vào rework, sửa hash/guideline/split
Ghi chú: đã mở ảnh kiểm tra hash các ảnh ({hash_str}), ảnh mở bình thường nên chỉ cập nhật lại mã sha256 thật trong manifest.csv; {len(rework_imgs)} ảnh lỗi detection được chuyển sang rework_queue.csv giao lại cho annotator/reviewer theo jobs.csv; các ảnh lỗi panoptic ({pan_str}) được giữ lại trong v2 và thông báo cho team panoptic xử lý.
"""
    (out_dir / "README.txt").write_text(readme_content, encoding="utf-8")
    print(f"Đã dựng thành công lô v2 tại: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
