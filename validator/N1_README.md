# N1 — Kiểm tra file và box detection

Người phụ trách: Lê Ngọc Dũng — 2A202603103 (Nhóm 044).

## Giao diện ghép nhóm

`rules_det.check_detection(delivery)` nhận đường dẫn delivery và trả list dict `{"type": ..., "target": ...}`. Không sửa input. Khi gọi `python validator/validate.py ...`, thư mục validator đã nằm trên đường dẫn import nên dùng `from rules_det import check_detection`.

Hiện chỉ có 5 loại N1: missing_image, wrong_hash, invalid_class, bbox_out_of_bounds, duplicate_box. N2/N3 chưa nối. Cờ --candidates hiện là khung rỗng, không được dùng làm bằng chứng đã đối chiếu ngữ nghĩa.

## Quy tắc

- Đọc manifest và tính SHA256 ảnh bằng byte. Ảnh thiếu thì bỏ qua hash/nhãn của ảnh đó.
- ID lớp hợp lệ lấy từ classes.txt; lớp phải là số nguyên.
- Tràn biên dùng dung sai 0.01, theo check_handover.py hiện có; chưa suy rằng mọi ngưỡng đều tối ưu trên đề ẩn.
- Khung lặp: cùng lớp; sai khác w/h tối đa 0.000002 do làm tròn; sai khác tâm trên mỗi trục tối đa 2.01 px (2 px cộng biên làm tròn). Kích thước ảnh lấy từ panoptic.json/images theo file_name, không phân tích ID từ tên ảnh.
- Ưu tiên nội bộ N1: missing_image > wrong_hash > invalid_class > bbox_out_of_bounds > duplicate_box. Khi ghép N2, lớp điều phối cần áp dụng ưu tiên chung đã thống nhất, gồm guideline_mismatch trước lỗi lớp/box; scene_leak giữ target là scene_id.

## Kết quả kiểm cục bộ

| Lô | Số mục manifest | Phát hiện N1 | Theo loại |
|---|---:|---:|---|
| r44007 | 250 | 7 | bbox_out_of_bounds: 1, duplicate_box: 2, invalid_class: 1, missing_image: 2, wrong_hash: 1 |
| r42 | 250 | 4 | bbox_out_of_bounds: 1, duplicate_box: 1, wrong_hash: 2 |
| clean | 50 | 0 | không có |

Đã kiểm 17 ca dữ liệu tạm về file, hash, lớp, bốn biên, dung sai, khung lặp, ưu tiên và không sửa đầu vào. Release ID và hash nhãn của r44007/clean khớp delivery đã tạo. Chưa đo F1 đầy đủ; không sử dụng đáp án _private hoặc phép thử điểm để suy ra từng lỗi.

## Chạy lại

```powershell
python -X utf8 validator/validate.py runs/r44007/delivery --out runs/r44007/rules_n1.json
python -X utf8 validator/validate.py runs/r42/delivery --out runs/r42/rules_n1.json
python -X utf8 validator/validate.py runs/clean/delivery --out runs/clean/rules_n1.json
```

Kết quả runs/ chỉ ở máy người chạy, không commit. Chưa push theo yêu cầu của người học.
