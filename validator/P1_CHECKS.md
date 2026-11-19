# P1 — Kiểm tra tích hợp N1 + N2, Nhóm 044

N1: Lê Ngọc Dũng; N2: Bùi Quang Huy. Giữ nguyên rules_meta.py của N2 và scripts/ của đề; chỉ nối check_meta vào validate.py.

## Kết quả tổng hợp

| Lô | Mục manifest | Lỗi luật phát hiện | F1 lỗi luật (scorer học viên) | F1 tổng (scorer học viên) |
|---|---:|---:|---:|---:|
| Seed 44007 | 250 | 12 | ≈ 1,0 | ≈ 0,7 |
| Seed 42 | 250 | 11 | ≈ 1,0 | ≈ 0,8 |
| Lô sạch, seed 5 | 50 | 0 | ≈ 1,0 | ≈ 1,0 |

F1 là số làm tròn một chữ số do script gốc in, không khẳng định giá trị chưa làm tròn bằng đúng 1. Lỗi gán nhãn của hai seed luyện tập hiện hiển thị `—` vì chưa nộp phát hiện ngữ nghĩa; chưa có N3 hoặc review. F1 ngữ nghĩa 1,0 của lô sạch là trường hợp không có lỗi/không báo lỗi, không chứng minh đã bắt được lỗi ngữ nghĩa.

Đạt mốc kiểm lỗi luật CP1 theo số liệu tự đo hiện tại; không phải xác nhận bàn giao v2 hoặc đề ẩn đã hoàn tất.

## Kiểm chứng tích hợp

- Gộp một lỗi ưu tiên cao nhất cho mỗi target ảnh theo quy ước nhóm; scene_leak giữ target cảnh riêng.
- Đã thử thứ tự ưu tiên, loại trùng, đảo thứ tự kết quả, dữ liệu rỗng và chặn loại lỗi ngữ nghĩa trong đầu ra lỗi luật.
- Hai tham số --out và --candidates phải là hai file khác nhau; không ghi đầu ra vào delivery.
- SHA256 của mọi file trong cả ba thư mục delivery được đối chiếu trước/sau, không thay đổi.
- Scorer chạy một lần cho mỗi danh sách trọn vẹn, chế độ học viên mặc định. Không dùng --truth-dir, không xem đáp án và không thử từng phần danh sách để dò lỗi.
- rules_meta.py của Huy không bị sửa. scripts/ và delivery/ được giữ nguyên.

## Chạy lại

```powershell
python -X utf8 validator/validate.py runs/r44007/delivery --out runs/r44007/errors.json
python -X utf8 scripts/score_findings.py --run runs/r44007
python -X utf8 validator/validate.py runs/r42/delivery --out runs/r42/errors.json
python -X utf8 scripts/score_findings.py --run runs/r42
python -X utf8 validator/validate.py runs/clean/delivery --out runs/clean/errors.json
python -X utf8 scripts/score_findings.py --run runs/clean
```

## Bản ghi chạy

# Kiểm tra tích hợp P1 — N1 + N2

Chỉ kiểm luật, chưa có review ngữ nghĩa N3. Scorer chạy chế độ học viên mặc định, không dùng --truth-dir; chỉ ghi số F1 đã làm tròn do script in.

## r44007

```text
Đã đọc manifest: 250 mục
Đã nối N1 + N2: đủ 9 loại lỗi luật, có ưu tiên và khử trùng.
Chưa nối N3: --candidates vẫn là khung, chưa đối chiếu ngữ nghĩa.
Số phát hiện: 12
Đã ghi: runs\r44007\errors.json
{'bbox_out_of_bounds': 1, 'duplicate_box': 2, 'guideline_mismatch': 1, 'invalid_category_id': 1, 'invalid_class': 1, 'missing_image': 2, 'scene_leak': 1, 'thing_overlap': 2, 'wrong_hash': 1}
# Tự đo — seed 44007

- Số lỗi nộp: 12
- F1 tổng ≈ 0.7 · lỗi luật ≈ 1.0 · lỗi gán nhãn ≈ —

Chỉ in làm tròn: dùng để biết validator tốt lên hay kém đi, không để dò từng ảnh.
```

Input delivery byte hashes unchanged.

## r42

```text
Đã đọc manifest: 250 mục
Đã nối N1 + N2: đủ 9 loại lỗi luật, có ưu tiên và khử trùng.
Chưa nối N3: --candidates vẫn là khung, chưa đối chiếu ngữ nghĩa.
Số phát hiện: 11
Đã ghi: runs\r42\errors.json
{'bbox_out_of_bounds': 1, 'duplicate_box': 1, 'guideline_mismatch': 2, 'invalid_category_id': 2, 'scene_leak': 1, 'thing_overlap': 2, 'wrong_hash': 2}
# Tự đo — seed 42

- Số lỗi nộp: 11
- F1 tổng ≈ 0.8 · lỗi luật ≈ 1.0 · lỗi gán nhãn ≈ —

Chỉ in làm tròn: dùng để biết validator tốt lên hay kém đi, không để dò từng ảnh.
```

Input delivery byte hashes unchanged.

## clean

```text
Đã đọc manifest: 50 mục
Đã nối N1 + N2: đủ 9 loại lỗi luật, có ưu tiên và khử trùng.
Chưa nối N3: --candidates vẫn là khung, chưa đối chiếu ngữ nghĩa.
Số phát hiện: 0
Đã ghi: runs\clean\errors.json
{}
# Tự đo — seed 5

- Số lỗi nộp: 0
- F1 tổng ≈ 1.0 · lỗi luật ≈ 1.0 · lỗi gán nhãn ≈ 1.0

Chỉ in làm tròn: dùng để biết validator tốt lên hay kém đi, không để dò từng ảnh.
```

Input delivery byte hashes unchanged.
