import argparse
import json
from pathlib import Path

from rules_det import check_detection


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

    # Không cho ghi kết quả vào bên trong lô v1.
    for output in outputs:
        if output.resolve().is_relative_to(delivery):
            parser.error("File kết quả phải nằm ngoài thư mục delivery")

    manifest = json.loads(
        (delivery / "manifest.json").read_text(encoding="utf-8")
    )

    # Hiện nối bốn kiểm tra N1; duplicate_box và N2/N3 sẽ bổ sung sau.
    errors = check_detection(delivery)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Sẽ nối phần sinh ứng viên của N3 tại đây.
    if args.candidates is not None:
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(
            json.dumps({"candidates": []}, indent=2),
            encoding="utf-8",
        )

    print(f"Đã đọc manifest: {len(manifest['items'])} mục")
    print("Đã kiểm N1: missing_image, wrong_hash, invalid_class, bbox_out_of_bounds.")
    print("Chưa kiểm duplicate_box và chưa nối N2/N3.")
    print(f"Số phát hiện: {len(errors)}")
    print(f"Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
    
