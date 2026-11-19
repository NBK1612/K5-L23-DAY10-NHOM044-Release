import argparse
import json
from pathlib import Path

from rules_det import check_detection
from rules_meta import check_meta

try:
    from crosscheck import find_candidates
except ImportError:
    from validator.crosscheck import find_candidates


# Quy ước ưu tiên của Nhóm 044; scene_leak được giữ riêng theo cảnh.
RULE_PRIORITY = {
    "missing_image": 0,
    "wrong_hash": 1,
    "guideline_mismatch": 2,
    "invalid_class": 3,
    "bbox_out_of_bounds": 4,
    "duplicate_box": 5,
    "invalid_category_id": 6,
    "thing_overlap": 7,
}


def merge_rule_findings(findings):
    """Mỗi ảnh giữ lỗi ưu tiên cao nhất; khử trùng cảnh độc lập với ảnh."""
    images = {}
    scenes = set()
    for finding in findings:
        kind, target = finding["type"], finding["target"]
        if kind == "scene_leak":
            scenes.add(target)
            continue
        if kind not in RULE_PRIORITY:
            raise ValueError(f"Không phải loại lỗi luật được hỗ trợ: {kind}")
        current = images.get(target)
        if current is None or RULE_PRIORITY[kind] < RULE_PRIORITY[current]:
            images[target] = kind

    pairs = [(kind, target) for target, kind in images.items()]
    pairs.extend(("scene_leak", target) for target in scenes)
    return [{"type": kind, "target": target} for kind, target in sorted(pairs)]


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra lô dataset")
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    outputs = [args.out]
    if args.candidates is not None:
        outputs.append(args.candidates)
    if len({output.resolve() for output in outputs}) != len(outputs):
        parser.error("--out và --candidates phải là hai file khác nhau")

    # Không cho ghi kết quả vào bên trong lô v1.
    for output in outputs:
        if output.resolve().is_relative_to(delivery):
            parser.error("File kết quả phải nằm ngoài thư mục delivery")

    manifest = json.loads(
        (delivery / "manifest.json").read_text(encoding="utf-8")
    )

    # N1 và N2 chỉ đọc delivery; gộp xong mới ghi một kết quả chung.
    errors = merge_rule_findings(
        check_detection(delivery) + check_meta(delivery)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Nối phần sinh ứng viên của N3 (Pha 2.2):
    candidates_count = 0
    if args.candidates is not None:
        candidates = find_candidates(delivery, rule_errors=errors)
        candidates_count = len(candidates)
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(
            json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Đã đọc manifest: {len(manifest['items'])} mục")
    print("Đã nối N1 + N2: đủ 9 loại lỗi luật, có ưu tiên và khử trùng.")
    if args.candidates is not None:
        print(f"Đã nối N3: sinh {candidates_count} ứng viên vào {args.candidates}")
    else:
        print("Không dùng cờ --candidates, chỉ xuất lỗi luật.")
    print(f"Số phát hiện lỗi luật: {len(errors)}")
    print(f"Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
    
