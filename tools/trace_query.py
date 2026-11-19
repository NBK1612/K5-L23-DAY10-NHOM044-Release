#!/usr/bin/env python3
"""Truy vết ticket của team AI (Pha 5 - N5).

Tự động phân tích:
  - incoming/ticket.md: lấy danh sách các cảnh (scene) bị báo lỗi
  - guideline/changelog.md: tìm lớp (class) bị đổi quy tắc từ guideline cũ sang mới
  - jobs.csv: xác định các job làm theo guideline cũ (thường là v1.2)
  - metadata.json & labels/: lọc các ảnh thuộc job cũ có chứa box detection của lớp đổi quy tắc
  
Đầu ra:
  - tier_a: ảnh trong cảnh được báo, job guideline cũ, có box detection lớp bị đổi
  - tier_b: ảnh ngoài cảnh đó nhưng cùng điều kiện (chưa ai báo)
  - notify: mã người gán (annotator) và người kiểm (reviewer) của các job liên quan

Cách dùng:
  python tools/trace_query.py runs/r44007/delivery --out runs/r44007/trace.json
  python tools/trace_query.py delivery --out submission/de_an/trace.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Truy vết ticket theo changelog và jobs")
    parser.add_argument("delivery", type=Path, help="Thư mục delivery cần truy vết")
    parser.add_argument("--out", type=Path, help="Đường dẫn file trace.json đầu ra")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    if not delivery.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục delivery: {delivery}")

    out_file = args.out.resolve() if args.out else (delivery.parent / "trace.json")

    ticket_file = delivery / "incoming" / "ticket.md"
    changelog_file = delivery / "guideline" / "changelog.md"
    jobs_file = delivery / "jobs.csv"
    meta_file = delivery / "metadata.json"
    classes_file = delivery / "classes.txt"
    labels_dir = delivery / "labels"

    # 1. Trích xuất cảnh bị báo từ ticket.md
    ticket_text = ticket_file.read_text(encoding="utf-8") if ticket_file.is_file() else ""
    scenes_reported = set(re.findall(r"scene_\w+", ticket_text))

    # 2. Đọc danh sách 80 lớp detection
    classes = [ln.split(" ", 1)[1].strip() for ln in classes_file.read_text(encoding="utf-8").splitlines()]
    class_to_id = {c: i for i, c in enumerate(classes)}

    # 3. Tìm lớp bị thay đổi quy tắc trong changelog.md
    changelog_text = changelog_file.read_text(encoding="utf-8") if changelog_file.is_file() else ""
    changed_classes = []
    for c in classes:
        # Kiểm tra lớp xuất hiện trong changelog dạng `name` hoặc Lớp name hoặc 'name'
        if f"`{c}`" in changelog_text or f"Lớp {c}" in changelog_text or f"lớp {c}" in changelog_text or f" {c}:" in changelog_text:
            changed_classes.append(c)

    changed_ids = {class_to_id[c] for c in changed_classes}

    # 4. Đọc jobs.csv để tìm guideline cũ
    jobs_list = read_csv(jobs_file)
    jobs = {row["job_id"]: row for row in jobs_list}
    all_guidelines = {row["guideline"] for row in jobs_list}
    # Guideline cũ thường có phiên bản thấp hơn (ví dụ v1.2 < v1.3)
    old_guideline = sorted(all_guidelines)[0]
    old_jobs = {jid for jid, j in jobs.items() if j["guideline"] == old_guideline}

    # 5. Đọc metadata.json
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))["items"]

    tier_a = []
    tier_b = []
    affected_jobs = set()

    for item in metadata:
        img_name = item["image"]
        job_id = item["job_id"]
        scene_id = item["scene_id"]

        # Chỉ xét job theo guideline cũ (lấy theo jobs.csv, không lấy theo metadata ảnh)
        if job_id not in old_jobs:
            continue

        lbl_path = labels_dir / f"{img_name}.txt"
        if not lbl_path.is_file():
            continue

        # Kiểm tra xem ảnh có ít nhất 1 box detection thuộc lớp bị đổi không
        has_changed_box = False
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cid = int(float(line.split()[0]))
                if cid in changed_ids:
                    has_changed_box = True
                    break

        if has_changed_box:
            affected_jobs.add(job_id)
            if scene_id in scenes_reported:
                tier_a.append(img_name)
            else:
                tier_b.append(img_name)

    tier_a.sort()
    tier_b.sort()

    # 6. notify: người gán và người kiểm của các job bị ảnh hưởng
    notify = sorted(
        {jobs[jid]["annotator"] for jid in affected_jobs}
        | {jobs[jid]["reviewer"] for jid in affected_jobs}
    )

    result = {
        "tier_a": tier_a,
        "tier_b": tier_b,
        "notify": notify,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Truy vết xong:")
    print(f"  - Tầng a (trong cảnh báo): {len(tier_a)} ảnh: {tier_a}")
    print(f"  - Tầng b (chưa báo): {len(tier_b)} ảnh: {tier_b}")
    print(f"  - Notify: {len(notify)} người: {notify}")
    print(f"=> Đã lưu tại: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
