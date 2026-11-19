"""Công cụ xem ảnh cho Pha 2.5 (N4). Được dùng Pillow; validator thì không.

Vẽ hai nguồn nhãn của cùng một ảnh lên ảnh gốc để người xem quyết định bên nào sai:
  - box detection (labels/*.txt, YOLO) màu đỏ, ghi "det:<lớp>";
  - viền segment panoptic (masks/*.png + panoptic.json) màu xanh, ghi "pan:<lớp>".
Các box/segment khớp với ứng viên (lớp class_det / class_pan) được tô đậm và có ô phóng to bên phải.

Cách dùng (chạy từ thư mục gốc repo):
  python tools/viz.py <thư_mục_delivery> --candidates <candidates.json> --out <thư_mục_ảnh_ra>
  python tools/viz.py <thư_mục_delivery> --images 000000001234 img_0007 --out <thư_mục_ảnh_ra>
  python tools/viz.py <thư_mục_delivery> --candidates <c.json> --assign review/assign.csv --who N1 --out <thư_mục>

Ví dụ:
  python tools/viz.py runs/r44007/delivery --candidates runs/r44007/candidates.json --out runs/r44007/viz
  python tools/viz.py delivery --candidates submission/de_an/candidates.json --out viz_de_an      (đề ẩn)

Đầu ra: <out>/<ảnh>.png cho từng ảnh và <out>/index.html để xem lần lượt trong trình duyệt.
Chỉ đọc delivery, không sửa gì trong đó. Không giả định tên ảnh dạng COCO.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    sys.exit("Cần Pillow cho công cụ xem ảnh: python -m pip install pillow")

DET_COLOR = (230, 40, 40)          # box detection
DET_FOCUS = (255, 150, 0)          # box detection khớp ứng viên
PAN_COLOR = (0, 200, 230)          # viền segment thing
PAN_FOCUS = (230, 0, 230)          # viền segment khớp ứng viên
CROWD_COLOR = (150, 150, 150)      # segment iscrowd
FULL_MAX = 720                     # cạnh dài nhất của ảnh toàn cảnh
ZOOM_MAX = 560                     # cạnh dài nhất của ô phóng to
HEADER_H = 74


# ---------------------------------------------------------------- đọc lô

def load_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class Delivery:
    """Đọc các file cần thiết của một lô (chỉ đọc)."""

    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["items"]
        self.classes = {}
        for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                cid, name = line.split(maxsplit=1)
                self.classes[int(cid)] = name.strip()
        pan = json.loads((root / "panoptic.json").read_text(encoding="utf-8"))
        self.cats = {c["id"]: c for c in pan["categories"]}
        self.ann_by_mask = {a["file_name"]: a for a in pan["annotations"]}

    def boxes(self, name: str, w: int, h: int) -> list[dict]:
        path = self.root / self.manifest[name]["label"]
        out = []
        if not path.is_file():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            out.append({
                "cls": self.classes.get(cid, f"cls{cid}?"),
                "box": ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h),
            })
        return out

    def segments(self, name: str) -> list[dict]:
        ann = self.ann_by_mask.get(Path(self.manifest[name]["mask"]).name, {})
        out = []
        for s in ann.get("segments_info", []):
            cat = self.cats.get(s["category_id"])
            x, y, bw, bh = s["bbox"]
            out.append({
                "id": s["id"],
                "cls": cat["name"] if cat else f"cat{s['category_id']}?",
                "thing": bool(cat and cat.get("isthing")),
                "crowd": bool(s.get("iscrowd")),
                "box": (x, y, x + bw, y + bh),
                "area": s.get("area", bw * bh),
            })
        return out


# ---------------------------------------------------------------- hình học

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def pick_focus(cand: dict, boxes: list[dict], segs: list[dict]):
    """Chọn box/segment liên quan tới ứng viên và vùng cần phóng to."""
    det_cls, pan_cls = cand.get("class_det"), cand.get("class_pan")
    f_boxes = [b for b in boxes if det_cls and b["cls"] == det_cls]
    f_segs = [s for s in segs if pan_cls and s["cls"] == pan_cls and s["thing"] and not s["crowd"]]
    kind = cand.get("type", "")

    if kind == "missing_box" and f_segs:
        # segment lớn nhất không có box nào chồng đáng kể
        def best_iou(s):
            return max((iou(s["box"], b["box"]) for b in boxes), default=0.0)
        seg = max(f_segs, key=lambda s: (best_iou(s) < 0.2, s["area"]))
        return [], [seg], seg["box"]

    if f_boxes and f_segs:
        box, seg = max(((b, s) for b in f_boxes for s in f_segs), key=lambda p: iou(p[0]["box"], p[1]["box"]))
        if iou(box["box"], seg["box"]) > 0:
            keep_boxes = [box]
            if kind == "instance_merge":
                keep_boxes = [b for b in f_boxes if iou(b["box"], seg["box"]) > 0]
            return keep_boxes, [seg], union_box([b["box"] for b in keep_boxes] + [seg["box"]])

    focus = [b["box"] for b in f_boxes] + [s["box"] for s in f_segs]
    return f_boxes, f_segs, union_box(focus) if focus else None


# ---------------------------------------------------------------- vẽ

def segment_ids(mask_path: Path, size) -> tuple[list[int], int, int]:
    img = Image.open(mask_path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    w, h = img.size
    raw = img.tobytes()
    ids = [raw[i] + raw[i + 1] * 256 + raw[i + 2] * 65536 for i in range(0, len(raw), 3)]
    return ids, w, h


def draw_outlines(canvas: Image.Image, ids: list[int], w: int, h: int, colors: dict[int, tuple]):
    """Tô các pixel biên của từng segment (pixel khác id với láng giềng phải/dưới)."""
    px = canvas.load()
    for y in range(h):
        row = y * w
        for x in range(w):
            sid = ids[row + x]
            right = ids[row + x + 1] if x + 1 < w else sid
            down = ids[row + w + x] if y + 1 < h else sid
            if sid != right or sid != down:
                for s in (sid, right, down):
                    if s in colors:
                        px[x, y] = colors[s]
                        break


def fill_segments(canvas: Image.Image, ids: list[int], w: int, h: int, focus_ids: set[int], color):
    if not focus_ids:
        return canvas
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    op = overlay.load()
    rgba = color + (70,)
    for i, sid in enumerate(ids):
        if sid in focus_ids:
            op[i % w, i // w] = rgba
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def render_zoom(base, mask_path, crop, boxes, segs, f_box_ids, f_seg_ids):
    """Phóng to vùng crop của ảnh gốc; vẽ viền 1 px ở độ phân giải đã phóng to."""
    piece = base.crop(crop)
    z = ZOOM_MAX / max(piece.size)
    size = (max(1, round(piece.width * z)), max(1, round(piece.height * z)))
    zoom = piece.resize(size, Image.LANCZOS)
    if mask_path.is_file():
        m = Image.open(mask_path).convert("RGB")
        if m.size != base.size:
            m = m.resize(base.size, Image.NEAREST)
        raw = m.crop(crop).resize(size, Image.NEAREST).tobytes()
        ids = [raw[i] + raw[i + 1] * 256 + raw[i + 2] * 65536 for i in range(0, len(raw), 3)]
        colors = {s["id"]: PAN_FOCUS if s["id"] in f_seg_ids else PAN_COLOR
                  for s in segs if s["thing"] and not s["crowd"]}
        draw_outlines(zoom, ids, size[0], size[1], colors)
    d = ImageDraw.Draw(zoom)
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        rect = ((x0 - crop[0]) * z, (y0 - crop[1]) * z, (x1 - crop[0]) * z, (y1 - crop[1]) * z)
        focus = id(b) in f_box_ids
        d.rectangle(rect, outline=DET_FOCUS if focus else DET_COLOR, width=2 if focus else 1)
    return zoom


def tag(draw, xy, text, color, font):
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle((l - 1, t - 1, r + 1, b + 1), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def render(dl: Delivery, name: str, cand: dict, out_dir: Path, fonts) -> Path | None:
    item = dl.manifest.get(name)
    if item is None or not (dl.root / item["image"]).is_file():
        print(f"  bỏ qua {name}: không có ảnh trong lô")
        return None
    base = Image.open(dl.root / item["image"]).convert("RGB")
    w, h = base.size
    boxes, segs = dl.boxes(name, w, h), dl.segments(name)
    f_boxes, f_segs, region = pick_focus(cand, boxes, segs)
    f_box_ids = {id(b) for b in f_boxes}
    f_seg_ids = {s["id"] for s in f_segs}

    canvas = base.copy()
    mask_path = dl.root / item["mask"]
    if mask_path.is_file():
        ids, mw, mh = segment_ids(mask_path, (w, h))
        canvas = fill_segments(canvas, ids, mw, mh, f_seg_ids, PAN_FOCUS)
        colors = {}
        for s in segs:
            if s["crowd"]:
                colors[s["id"]] = CROWD_COLOR
            elif s["thing"]:
                colors[s["id"]] = PAN_FOCUS if s["id"] in f_seg_ids else PAN_COLOR
        draw_outlines(canvas, ids, mw, mh, colors)

    small, big = fonts
    d = ImageDraw.Draw(canvas)
    for s in segs:
        if s["thing"]:
            label = f"pan:{s['cls']}" + (" (crowd)" if s["crowd"] else "")
            col = PAN_FOCUS if s["id"] in f_seg_ids else (CROWD_COLOR if s["crowd"] else PAN_COLOR)
            tag(d, (s["box"][0] + 2, s["box"][3] - 14), label, col, small)
    for b in boxes:
        focus = id(b) in f_box_ids
        d.rectangle(b["box"], outline=DET_FOCUS if focus else DET_COLOR, width=3 if focus else 1)
        tag(d, (b["box"][0] + 2, b["box"][1] + 1), f"det:{b['cls']}", DET_FOCUS if focus else DET_COLOR, small)

    # ảnh toàn cảnh
    scale = min(1.0, FULL_MAX / max(w, h)) if max(w, h) > FULL_MAX else FULL_MAX / max(w, h)
    full = canvas.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    if region:
        fd = ImageDraw.Draw(full)
        fd.rectangle([v * scale for v in region], outline=(255, 255, 0), width=2)

    # ô phóng to quanh vùng nghi ngờ: ảnh gốc sạch, chỉ viền mảnh, không chữ (để không che vật)
    zoom = None
    if region:
        x0, y0, x1, y1 = region
        mx, my = max(16, (x1 - x0) * 0.25), max(16, (y1 - y0) * 0.25)
        crop = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx) + 1), min(h, int(y1 + my) + 1))
        zoom = render_zoom(base, mask_path, crop, boxes, segs, f_box_ids, f_seg_ids)

    width = full.width + (zoom.width + 10 if zoom else 0)
    height = HEADER_H + max(full.height, zoom.height if zoom else 0)
    sheet = Image.new("RGB", (max(width, 900), height), (25, 25, 25))
    sheet.paste(full, (0, HEADER_H))
    if zoom:
        sheet.paste(zoom, (full.width + 10, HEADER_H))
    sd = ImageDraw.Draw(sheet)
    title = f"{name}   |   nghi: {cand.get('type', '-')}   |   det: {cand.get('class_det') or '-'}   pan: {cand.get('class_pan') or '-'}   |   {w}x{h}"
    sd.text((8, 6), title, fill=(255, 255, 255), font=big)
    sd.text((8, 30), str(cand.get("reason", ""))[:160], fill=(220, 220, 220), font=small)
    sd.text((8, 52), "Đỏ = box detection · Xanh = viền panoptic · Cam/Tím = khớp ứng viên · Vàng = vùng phóng to · Xám = crowd",
            fill=(170, 170, 170), font=small)

    out = out_dir / f"{name}.png"
    sheet.save(out)
    return out


# ---------------------------------------------------------------- chọn ảnh

def load_jobs(args, dl: Delivery) -> list[dict]:
    cands = []
    if args.candidates:
        data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        cands = data["candidates"] if isinstance(data, dict) else data
    if args.images:
        known = {c["image"] for c in cands}
        cands += [{"image": n, "type": "xem", "reason": "xem thủ công"} for n in args.images if n not in known]
    if args.assign:
        with open(args.assign, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        mine = {r["image"].strip() for r in rows if not args.who or args.who.lower() in r.get("nguoi_xem", "").lower()}
        cands = [c for c in cands if c["image"] in mine]
    if not cands:
        sys.exit("Không có ảnh nào để vẽ: truyền --candidates và/hoặc --images.")
    unknown = [c["image"] for c in cands if c["image"] not in dl.manifest]
    if unknown:
        print(f"Cảnh báo: {len(unknown)} ảnh không có trong manifest: {unknown[:5]}")
    return cands


def write_index(out_dir: Path, done: list[tuple[dict, Path]]):
    rows = []
    for cand, path in done:
        rows.append(
            f"<h3>{html.escape(cand['image'])} — {html.escape(str(cand.get('type', '')))}</h3>"
            f"<p>{html.escape(str(cand.get('reason', '')))}</p>"
            f"<a href='{path.name}'><img src='{path.name}' style='max-width:100%'></a><hr>"
        )
    page = ("<!DOCTYPE html><meta charset='utf-8'><title>viz</title>"
            "<body style='background:#111;color:#eee;font-family:sans-serif'>"
            f"<p>{len(done)} ảnh. Ghi kết luận vào review/&lt;tên&gt;.csv: image,type,verdict,note</p>"
            + "".join(rows) + "</body>")
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Vẽ box detection và viền panoptic lên ảnh để review")
    ap.add_argument("delivery", type=Path)
    ap.add_argument("--candidates", help="candidates.json do validator sinh ra")
    ap.add_argument("--images", nargs="*", help="tên ảnh (không đuôi) muốn xem thêm")
    ap.add_argument("--assign", help="review/assign.csv: chỉ vẽ ảnh được giao")
    ap.add_argument("--who", help="lọc theo cột nguoi_xem trong --assign, ví dụ N1")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    delivery = args.delivery.resolve()
    out_dir = args.out.resolve()
    if out_dir == delivery or delivery in out_dir.parents:
        ap.error("--out phải nằm ngoài thư mục delivery")
    dl = Delivery(delivery)
    jobs = load_jobs(args, dl)
    out_dir.mkdir(parents=True, exist_ok=True)
    fonts = (load_font(13), load_font(17))

    done = []
    for i, cand in enumerate(jobs, 1):
        path = render(dl, cand["image"], cand, out_dir, fonts)
        if path:
            done.append((cand, path))
            print(f"[{i}/{len(jobs)}] {path.name}  ({cand.get('type', '')})")
    write_index(out_dir, done)
    print(f"Xong {len(done)} ảnh. Mở {out_dir / 'index.html'} để xem lần lượt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
