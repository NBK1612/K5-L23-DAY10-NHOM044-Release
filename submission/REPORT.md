# Báo cáo nhóm <<XX>>

## 1. Số liệu luyện tập
- F1 theo seed (rule / cross_check / overall), số làm tròn do `score_findings.py` in, mỗi danh sách trọn vẹn chạy một lần:

  | Lô | Lỗi luật báo | rule | cross_check | overall | Ghi chú |
  |---|---:|---:|---:|---:|---|
  | 44007 (chỉ lỗi luật, P1) | 12 | ≈ 1.0 | — | ≈ 0.7 | chưa có lỗi gán nhãn |
  | 42 (chỉ lỗi luật, P1) | 11 | ≈ 1.0 | — | ≈ 0.8 | chưa có lỗi gán nhãn |
  | clean, seed 5, 50 ảnh | 0 | ≈ 1.0 | — | ≈ 1.0 | lô không gài lỗi, validator không báo nhầm |
  | 44007 (sau review, P2) | <<>> | <<>> | <<>> | <<>> | <<>> |

  Lỗi luật seed 44007: 2 missing_image, 1 wrong_hash, 1 invalid_class, 1 bbox_out_of_bounds, 2 duplicate_box, 1 guideline_mismatch, 1 scene_leak, 1 invalid_category_id, 2 thing_overlap. Lô sạch 0 lỗi cho thấy `thing_overlap` (so `area` với số pixel đếm từ mask) không cần dung sai: ở dữ liệu gốc hai số bằng nhau tuyệt đối.
- Số ứng viên trên lô sạch (`--no-faults`): <<...>>; hàng đợi trên seed luyện tập: <<... ảnh>>
- Ngưỡng đã chọn và cách chọn (không dùng `_private`): <<...>>

## 2. Vì sao báo hết ứng viên làm F1 giảm?
<<Trả lời bằng số của nhóm.>>

## 3. Quyết định và lineage
<<QC v1 a/b, kết luận, vì sao R-021 phải xem lại hoặc train lại.>>

## 4. Hash chứng minh gì, không chứng minh gì?
<<>>

## 5. Truy vết
<<Tiêu chí tầng a/b, vì sao lấy guideline theo job.>>

## 6. Bug hoặc điểm chưa rõ trong lab (nếu có)
**Cùng seed nhưng `release_id` khác nhau tuỳ phiên bản Python** (Huy, N2 phát hiện).
- Seed 44007 với Python 3.12 (zlib 1.3.1) ra `lab250_e9b358f6faee`; với Python 3.14/3.15 trên Windows (zlib-ng) ra `lab250_050cd73ecce9`.
- So hai lô: ảnh, nhãn, `metadata.json` và danh sách lỗi gài giống hệt; chỉ 241/250 mask PNG khác sha256. Giải nén thử 40 mask thì pixel giống hệt: `pngutil.write_rgb` nén bằng `zlib`, zlib-ng nén ra byte khác, nên `mask_sha256` trong `manifest.json` và `release_id` đổi.
- Hệ quả: các máy trong nhóm không cùng `release_id`; nếu dựng v2 ở máy khác phiên bản thì hash mask lệch.
- Nhóm xử lý: cả nhóm chạy mọi script bằng `py -3.12`. Không sửa `scripts/`.
