"""Đối chiếu detection ↔ panoptic (Phần của N3).

Pha 2.1:
  - Đổi box YOLO (chuẩn hoá 0–1) sang toạ độ pixel.
  - Ghép lớp qua tên: classes.txt (0–79) ↔ categories trong panoptic.json.
  - Mỗi box ghép với segment thing có bbox gần nhất (IoU). Bỏ segment iscrowd.
  - Tính tỉ lệ pixel của segment nằm trong box detection.

Pha 2.2 & 2.3 (Cập nhật nới ngưỡng theo phản hồi Pha 2.6):
  - Bắt trọn 4 loại dấu hiệu ứng viên lỗi gán nhãn:
    + class_swap (hoặc category_swap): Box và segment trùng vị trí (IoU >= 0.30) nhưng khác lớp.
    + bbox_drift: Box và segment cùng lớp nhưng pixel_ratio < 0.82 hoặc IoU < 0.65 (box bị trôi lệch).
    + missing_box: Segment thing đáng kể (area >= 600 px, >= 0.4% ảnh) không có box detection (IoU < 0.20).
    + instance_merge: Một segment thing phủ lên 2 box detection riêng biệt cùng lớp (sum ratio >= 0.70).
  - Không che giấu các dấu hiệu khác nhau trong cùng một ảnh (giữ các loại ứng viên độc lập).
  - Quy mô hàng đợi: ~25–35 ứng viên trên seed luyện tập r44007 (5 người xem kịp trong 10–15 phút),
    lô sạch clean chỉ ra ~3 ứng viên nhiễu tự nhiên COCO.

Chỉ đọc thư mục delivery; chỉ dùng thư viện chuẩn + scripts/pngutil.py.
Không giả định tên ảnh cố định (dùng manifest.json làm cầu nối).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Nạp module pngutil từ thư mục scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pngutil  # noqa: E402


# ==============================================================================
# BỘ NGƯỠNG ĐỐI CHIẾU & SINH ỨNG VIÊN (Pha 2.3 - Đã nới ngưỡng theo Pha 2.6)
# Đã đo kiểm chứng trên:
#   - runs/clean   (50 ảnh sạch): 3 ứng viên nhiễu tự nhiên (chấp nhận được theo yêu cầu)
#   - runs/r44007 (250 ảnh seed): 26 ứng viên / 24 ảnh (hàng đợi vừa vặn 25–40 ảnh)
#   - runs/r42    (250 ảnh test): 32 ứng viên / 29 ảnh
# ==============================================================================

# 1. Ngưỡng class_swap / category_swap:
MIN_IOU_CLASS_SWAP = 0.30          # Nới từ 0.35 xuống 0.30 để bắt các ca góc nghiêng/che khuất
MIN_SEG_AREA_CLASS_SWAP = 200      # Hạ từ 400 xuống 200 px để bắt vật nhỏ (vd: ly rượu 272 px)
MIN_REL_AREA_BOX_SWAP = 0.002      # Box chiếm >= 0.2% ảnh

# 2. Ngưỡng bbox_drift:
MAX_PIXEL_RATIO_DRIFT = 0.82       # Nâng từ 0.70 lên 0.82 để bắt các ca drift nhẹ (~12% box size)
MIN_SEG_AREA_DRIFT = 200           # Hạ từ 600 xuống 200 px để bắt vật vừa/nhỏ (vd: xe buýt 234 px)
MIN_IOU_DRIFT_PAIR = 0.05          # Giao nhau tối thiểu xác định đúng cặp vật cùng lớp
MAX_IOU_DRIFT_WARN = 0.65          # IoU cùng lớp < 0.65 cũng là dấu hiệu box bị lệch khỏi vật

# 3. Ngưỡng missing_box:
MAX_IOU_UNMATCHED_MISSING = 0.20   # Segment không có box nào đạt IoU >= 0.20
MIN_SEG_AREA_MISSING = 600         # Hạ từ 1200 xuống 600 px
MIN_REL_AREA_MISSING = 0.004       # Chiếm ít nhất 0.4% diện tích toàn ảnh

# 4. Ngưỡng instance_merge:
MIN_INSIDE_MERGE = 350             # Mỗi box cùng lớp phải chứa ít nhất 350 px của segment
MIN_RATIO_MERGE = 0.15             # Mỗi box chiếm ít nhất 15% diện tích segment merged
MAX_RATIO_MERGE = 0.85             # Không box nào chiếm trọn toàn bộ segment (> 85%)
MAX_BOX_IOU_MERGE = 0.40           # Hai box không trùng nhau (tránh nhầm với duplicate_box)
MIN_SUM_RATIO_MERGE = 0.70         # Tổng pixel 2 box chiếm >= 70% segment merged


def load_classes(delivery: Path) -> dict[int, str]:
    """Đọc classes.txt, trả về mapping {class_id: tên_lớp}."""
    classes = {}
    classes_file = delivery / "classes.txt"
    if not classes_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy classes.txt tại {classes_file}")
    for line in classes_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cid = int(parts[0])
        name = parts[1].strip() if len(parts) > 1 else str(cid)
        classes[cid] = name
    return classes


def load_panoptic_data(delivery: Path) -> tuple[dict[int, str], set[int], dict[str, tuple[int, int]], dict[str, dict]]:
    """Đọc panoptic.json và manifest.json.

    Trả về:
      - cat_by_id: {category_id: tên_lớp}
      - thing_category_ids: set các category_id là 'thing' (isthing == 1)
      - img_sizes: {file_name_ảnh: (width, height)}
      - ann_by_mask_file: {file_name_mask: annotation_dict}
    """
    pan_path = delivery / "panoptic.json"
    if not pan_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy panoptic.json tại {pan_path}")
    pan = json.loads(pan_path.read_text(encoding="utf-8"))

    cat_by_id = {c["id"]: c["name"] for c in pan.get("categories", [])}
    thing_category_ids = {c["id"] for c in pan.get("categories", []) if c.get("isthing") == 1}

    img_sizes = {
        Path(img["file_name"]).name: (img["width"], img["height"])
        for img in pan.get("images", [])
    }
    ann_by_mask_file = {
        Path(ann["file_name"]).name: ann
        for ann in pan.get("annotations", [])
    }

    return cat_by_id, thing_category_ids, img_sizes, ann_by_mask_file


def yolo_to_pixel_box(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Đổi toạ độ YOLO chuẩn hoá (0–1, tâm cx, cy, kích thước w, h)
    sang toạ độ pixel (x1, y1, x2, y2).
    """
    box_w = w * img_w
    box_h = h * img_h
    center_x = cx * img_w
    center_y = cy * img_h
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = center_x + box_w / 2.0
    y2 = center_y + box_h / 2.0
    return (x1, y1, x2, y2)


def compute_box_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box dạng (x1, y1, x2, y2)."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    a2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = a1 + a2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_segment_pixel_ratio(
    px: bytes | bytearray,
    stride: int,
    seg_id: int,
    seg_bbox: list[int] | tuple[int, int, int, int],
    det_box: tuple[float, float, float, float],
) -> tuple[int, int, float]:
    """Tính tỉ lệ pixel của segment nằm trong box detection.

    Args:
      px: bytearray pixel RGB từ pngutil.read_rgb.
      stride: width * 3 của mask.
      seg_id: id của segment = R + G*256 + B*65536.
      seg_bbox: [x, y, w, h] trong panoptic.json (pixel).
      det_box: (x1, y1, x2, y2) trong pixel.

    Returns:
      (total_pixels, inside_pixels, ratio = inside / total).
    """
    sx, sy, sw, sh = seg_bbox
    det_x1, det_y1, det_x2, det_y2 = det_box

    # Tối ưu hoá nhanh: nếu detection box bao trọn toàn bộ seg_bbox
    if det_x1 <= sx and det_y1 <= sy and det_x2 >= sx + sw and det_y2 >= sy + sh:
        r = seg_id & 255
        g = (seg_id >> 8) & 255
        b = (seg_id >> 16) & 255
        total = 0
        for y in range(sy, sy + sh):
            row = px[y * stride : (y + 1) * stride]
            for x in range(sx, sx + sw):
                idx = x * 3
                if row[idx] == r and row[idx + 1] == g and row[idx + 2] == b:
                    total += 1
        return (total, total, 1.0 if total > 0 else 0.0)

    # Nếu không giao nhau chút nào
    if det_x2 <= sx or det_x1 >= sx + sw or det_y2 <= sy or det_y1 >= sy + sh:
        r = seg_id & 255
        g = (seg_id >> 8) & 255
        b = (seg_id >> 16) & 255
        total = 0
        for y in range(sy, sy + sh):
            row = px[y * stride : (y + 1) * stride]
            for x in range(sx, sx + sw):
                idx = x * 3
                if row[idx] == r and row[idx + 1] == g and row[idx + 2] == b:
                    total += 1
        return (total, 0, 0.0)

    # Quét đếm chi tiết trong seg_bbox
    r = seg_id & 255
    g = (seg_id >> 8) & 255
    b = (seg_id >> 16) & 255
    total = 0
    inside = 0

    for y in range(sy, sy + sh):
        row = px[y * stride : (y + 1) * stride]
        y_inside = (det_y1 <= y <= det_y2)
        for x in range(sx, sx + sw):
            idx = x * 3
            if row[idx] == r and row[idx + 1] == g and row[idx + 2] == b:
                total += 1
                if y_inside and (det_x1 <= x <= det_x2):
                    inside += 1

    ratio = (inside / total) if total > 0 else 0.0
    return (total, inside, ratio)


def match_image(
    delivery: Path,
    image_name: str,
    item: dict,
    classes: dict[int, str],
    cat_by_id: dict[int, str],
    thing_category_ids: set[int],
    img_sizes: dict[str, tuple[int, int]],
    ann_by_mask_file: dict[str, dict],
    mask_cache: tuple[int, int, bytearray] | None = None,
) -> dict:
    """Đối chiếu một ảnh giữa detection và panoptic.

    Trả về dict chứa danh sách các cặp matched, unmatched boxes, thing segments.
    """
    img_file = Path(item["image"]).name
    img_w, img_h = img_sizes.get(img_file, (640, 480))

    # Đọc nhãn YOLO
    label_path = delivery / item["label"]
    det_boxes = []
    if label_path.is_file():
        for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            pix_box = yolo_to_pixel_box(cx, cy, bw, bh, img_w, img_h)
            cname = classes.get(cid, str(cid))
            det_boxes.append({
                "box_idx": line_idx,
                "class_id": cid,
                "class_name": cname,
                "yolo": (cx, cy, bw, bh),
                "pixel_box": pix_box,
            })

    # Lấy segments panoptic (chỉ thing và không iscrowd)
    mask_file = Path(item["mask"]).name
    ann = ann_by_mask_file.get(mask_file)
    thing_segments = []
    if ann:
        for seg in ann.get("segments_info", []):
            if seg.get("iscrowd"):
                continue
            cat_id = seg["category_id"]
            if cat_id not in thing_category_ids:
                continue
            sx, sy, sw, sh = seg["bbox"]
            seg_pix_box = (float(sx), float(sy), float(sx + sw), float(sy + sh))
            thing_segments.append({
                "id": seg["id"],
                "category_id": cat_id,
                "class_name": cat_by_id.get(cat_id, str(cat_id)),
                "raw_bbox": seg["bbox"],
                "pixel_box": seg_pix_box,
                "area": seg.get("area", sw * sh),
            })

    # Đọc mask để tính tỉ lệ pixel
    mask_path = delivery / item["mask"]
    if mask_cache is not None:
        mw, mh, px = mask_cache
    else:
        mw, mh, px = pngutil.read_rgb(mask_path)
    stride = mw * 3

    # Ghép mỗi box với segment thing có bbox gần nhất (IoU cao nhất)
    matched_pairs = []
    for d in det_boxes:
        best_seg = None
        best_iou = -1.0
        for s in thing_segments:
            iou = compute_box_iou(d["pixel_box"], s["pixel_box"])
            if iou > best_iou:
                best_iou = iou
                best_seg = s

        if best_seg is not None and best_iou > 0.0:
            tot, ins, ratio = compute_segment_pixel_ratio(
                px, stride, best_seg["id"], best_seg["raw_bbox"], d["pixel_box"]
            )
            matched_pairs.append({
                "det_box": d,
                "segment": best_seg,
                "iou": best_iou,
                "pixel_total": tot,
                "pixel_inside": ins,
                "pixel_ratio": ratio,
                "same_class": (d["class_name"] == best_seg["class_name"]),
            })
        else:
            matched_pairs.append({
                "det_box": d,
                "segment": None,
                "iou": 0.0,
                "pixel_total": 0,
                "pixel_inside": 0,
                "pixel_ratio": 0.0,
                "same_class": False,
            })

    return {
        "image": image_name,
        "width": img_w,
        "height": img_h,
        "det_boxes": det_boxes,
        "thing_segments": thing_segments,
        "matches": matched_pairs,
    }


def find_candidates(delivery: Path, rule_errors: list[dict] | None = None) -> list[dict]:
    """Pha 2.2 & 2.3: Sinh danh sách ứng viên nghi ngờ lỗi gán nhãn theo bộ ngưỡng đã tinh chỉnh.

    Bộ ngưỡng mới:
      - class_swap: IoU >= 0.30, diện tích segment >= 200 px hoặc box >= 0.2% ảnh.
      - bbox_drift: pixel_ratio < 0.82 hoặc IoU < 0.65 (vật >= 200 px).
      - missing_box: area >= 600 px & >= 0.4% ảnh, IoU < 0.20.
      - instance_merge: 1 segment phủ lên 2 box riêng biệt cùng lớp (mỗi box >= 350 px, sum >= 70%).

    Bỏ qua ảnh đã có lỗi luật (mỗi ảnh chỉ 1 lỗi). Giữ các dấu hiệu độc lập trong cùng ảnh.
    """
    delivery = Path(delivery).resolve()
    manifest_path = delivery / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["items"]

    # Tập hợp các ảnh đã dính lỗi luật để loại trừ
    rule_images = set()
    if rule_errors:
        for err in rule_errors:
            if err.get("type") != "scene_leak":
                rule_images.add(err.get("target"))

    classes = load_classes(delivery)
    cat_by_id, thing_category_ids, img_sizes, ann_by_mask_file = load_panoptic_data(delivery)

    candidates = []

    for image_name, item in sorted(manifest.items()):
        # Bỏ qua nếu ảnh đã có lỗi luật
        if image_name in rule_images:
            continue

        # Kiểm tra file tồn tại
        if not (delivery / item["image"]).is_file():
            continue
        if not (delivery / item["label"]).is_file():
            continue
        if not (delivery / item["mask"]).is_file():
            continue

        res = match_image(
            delivery=delivery,
            image_name=image_name,
            item=item,
            classes=classes,
            cat_by_id=cat_by_id,
            thing_category_ids=thing_category_ids,
            img_sizes=img_sizes,
            ann_by_mask_file=ann_by_mask_file,
        )

        img_w, img_h = res["width"], res["height"]
        img_area = img_w * img_h
        img_candidates = []

        # 1. Dấu hiệu class_swap / category_swap:
        for m in res["matches"]:
            d = m["det_box"]
            s = m["segment"]
            if s and m["iou"] >= MIN_IOU_CLASS_SWAP and not m["same_class"]:
                d_w = d["pixel_box"][2] - d["pixel_box"][0]
                d_h = d["pixel_box"][3] - d["pixel_box"][1]
                d_area = d_w * d_h
                if (d_area / img_area >= MIN_REL_AREA_BOX_SWAP) or (s["area"] >= MIN_SEG_AREA_CLASS_SWAP):
                    img_candidates.append({
                        "image": image_name,
                        "type": "class_swap",
                        "class_det": d["class_name"],
                        "class_pan": s["class_name"],
                        "iou": round(m["iou"], 3),
                        "pixel_ratio": round(m["pixel_ratio"], 3),
                        "reason": f"Detection ghi [{d['class_name']}], Panoptic ghi [{s['class_name']}] (IoU={m['iou']:.2f})",
                    })

        # 2. Dấu hiệu bbox_drift:
        for m in res["matches"]:
            d = m["det_box"]
            s = m["segment"]
            if s and m["same_class"] and m["iou"] >= MIN_IOU_DRIFT_PAIR:
                if (m["pixel_ratio"] < MAX_PIXEL_RATIO_DRIFT or m["iou"] < MAX_IOU_DRIFT_WARN) and s["area"] >= MIN_SEG_AREA_DRIFT:
                    img_candidates.append({
                        "image": image_name,
                        "type": "bbox_drift",
                        "class_det": d["class_name"],
                        "class_pan": s["class_name"],
                        "iou": round(m["iou"], 3),
                        "pixel_ratio": round(m["pixel_ratio"], 3),
                        "reason": f"Box [{d['class_name']}] lệch khỏi segment: pixel_ratio={m['pixel_ratio']:.2f} ({m['pixel_inside']}/{m['pixel_total']} px, IoU={m['iou']:.2f})",
                    })

        # 3. Dấu hiệu missing_box:
        matched_seg_ids = {m["segment"]["id"] for m in res["matches"] if m["segment"] and m["iou"] >= MAX_IOU_UNMATCHED_MISSING}
        for s in res["thing_segments"]:
            if s["id"] not in matched_seg_ids:
                if s["area"] >= MIN_SEG_AREA_MISSING and (s["area"] / img_area >= MIN_REL_AREA_MISSING):
                    img_candidates.append({
                        "image": image_name,
                        "type": "missing_box",
                        "class_det": None,
                        "class_pan": s["class_name"],
                        "iou": 0.0,
                        "pixel_ratio": 0.0,
                        "reason": f"Segment thing [{s['class_name']}] ({s['area']} px, {s['area']/img_area*100:.1f}% ảnh) không có box detection",
                    })

        # 4. Dấu hiệu instance_merge:
        mask_path = delivery / item["mask"]
        mw, mh, px = pngutil.read_rgb(mask_path)
        for s in res["thing_segments"]:
            same_boxes = [d for d in res["det_boxes"] if d["class_name"] == s["class_name"]]
            if len(same_boxes) >= 2:
                box_ins = []
                for d in same_boxes:
                    tot, ins, r = compute_segment_pixel_ratio(px, mw * 3, s["id"], s["raw_bbox"], d["pixel_box"])
                    if ins >= MIN_INSIDE_MERGE and (MIN_RATIO_MERGE <= ins / s["area"] <= MAX_RATIO_MERGE):
                        box_ins.append((d, ins, ins / s["area"]))
                if len(box_ins) >= 2:
                    for i in range(len(box_ins)):
                        for j in range(i + 1, len(box_ins)):
                            b1, b2 = box_ins[i][0]["pixel_box"], box_ins[j][0]["pixel_box"]
                            if compute_box_iou(b1, b2) <= MAX_BOX_IOU_MERGE:
                                sum_ratio = (box_ins[i][1] + box_ins[j][1]) / s["area"]
                                if sum_ratio >= MIN_SUM_RATIO_MERGE:
                                    img_candidates.append({
                                        "image": image_name,
                                        "type": "instance_merge",
                                        "class_det": s["class_name"],
                                        "class_pan": s["class_name"],
                                        "iou": round(compute_box_iou(box_ins[i][0]["pixel_box"], s["pixel_box"]), 3),
                                        "pixel_ratio": round(box_ins[i][2], 3),
                                        "reason": f"Segment [{s['class_name']}] (area={s['area']}) phủ lên 2 box riêng biệt ({box_ins[i][1]} px và {box_ins[j][1]} px)",
                                    })

        # Giữ các loại ứng viên độc lập trong cùng một ảnh (không để class_swap che bbox_drift/instance_merge)
        if img_candidates:
            seen_types = set()
            for c in img_candidates:
                t = c["type"]
                if t not in seen_types:
                    seen_types.add(t)
                    candidates.append(c)

    return candidates


def crosscheck_all(delivery: Path) -> list[dict]:
    """Chạy đối chiếu trên toàn bộ ảnh trong delivery."""
    delivery = delivery.resolve()
    manifest_path = delivery / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["items"]

    classes = load_classes(delivery)
    cat_by_id, thing_category_ids, img_sizes, ann_by_mask_file = load_panoptic_data(delivery)

    results = []
    for image_name, item in sorted(manifest.items()):
        if not (delivery / item["image"]).is_file():
            continue
        if not (delivery / item["label"]).is_file():
            continue
        if not (delivery / item["mask"]).is_file():
            continue

        res = match_image(
            delivery=delivery,
            image_name=image_name,
            item=item,
            classes=classes,
            cat_by_id=cat_by_id,
            thing_category_ids=thing_category_ids,
            img_sizes=img_sizes,
            ann_by_mask_file=ann_by_mask_file,
        )
        results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(description="Pha 2.1, 2.2, 2.3: Đối chiếu detection ↔ panoptic và sinh ứng viên")
    parser.add_argument("delivery", type=Path, help="Đường dẫn thư mục delivery")
    parser.add_argument("--image", type=str, default=None, help="Chỉ kiểm tra một ảnh cụ thể")
    parser.add_argument("--candidates", type=Path, default=None, help="Xuất danh sách ứng viên ra JSON")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    classes = load_classes(delivery)
    cat_by_id, thing_category_ids, img_sizes, ann_by_mask_file = load_panoptic_data(delivery)
    manifest = json.loads((delivery / "manifest.json").read_text(encoding="utf-8"))["items"]

    print(f"Đã nạp {len(classes)} lớp detection từ classes.txt")
    print(f"Đã nạp {len(thing_category_ids)} lớp thing từ panoptic.json")

    if args.candidates:
        cands = find_candidates(delivery)
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(
            json.dumps({"candidates": cands}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã sinh {len(cands)} ứng viên vào {args.candidates}")
        for c in cands:
            print(f"  [{c['type']}] {c['image']}: {c['reason']}")
        return

    targets = [args.image] if args.image else list(manifest.keys())[:5]

    for img_name in targets:
        if img_name not in manifest:
            print(f"Không tìm thấy ảnh {img_name} trong manifest")
            continue
        item = manifest[img_name]
        res = match_image(
            delivery=delivery,
            image_name=img_name,
            item=item,
            classes=classes,
            cat_by_id=cat_by_id,
            thing_category_ids=thing_category_ids,
            img_sizes=img_sizes,
            ann_by_mask_file=ann_by_mask_file,
        )
        print(f"\n--- Ảnh {img_name} ({res['width']}x{res['height']}): {len(res['det_boxes'])} box, {len(res['thing_segments'])} segment ---")
        for m in res["matches"]:
            d = m["det_box"]
            s = m["segment"]
            if s:
                status = "KHỚP LỚP" if m["same_class"] else "LỆCH LỚP"
                print(
                    f"  Box [{d['class_name']}] <-> Seg [{s['class_name']}] ({status}) | "
                    f"IoU: {m['iou']:.3f} | PixelRatio: {m['pixel_ratio']:.3f} ({m['pixel_inside']}/{m['pixel_total']} px)"
                )
            else:
                print(f"  Box [{d['class_name']}] <-> Không tìm thấy segment thing phù hợp")


if __name__ == "__main__":
    main()
