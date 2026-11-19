import argparse
import json
from pathlib import Path


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

    # Sẽ nối các hàm kiểm N1 và N2 tại đây.
    errors = []

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
    print("Đây là khung điều phối, chưa thực hiện kiểm lỗi.")
    print(f"Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
    