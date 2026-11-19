Release: lab250_e9b358f6faee_v2
Parent: lab250_e9b358f6faee
Số ảnh: 236
Số nhãn: 1641
Guideline: v1.2 (J01-J04), v1.3 (J05-J10)
QC v1: 21/250 = 8.4%
Kết luận v1: Hold — vượt ngưỡng 2% (21/250 = 8.4%)
Kết luận v2: Release — đã tách 14 ảnh vào rework, sửa hash/guideline/split
Ghi chú: (1) Hash: ảnh 000000023751 (wrong_hash) đã mở kiểm, ảnh nguyên vẹn đúng nội dung nên giữ trong v2 và ghi sha256 thật vào manifest.csv. (2) Guideline: ảnh 000000015956 ghi v1.2 nhưng thuộc job J07 dùng v1.3; truyvet.csv lấy guideline theo job (v1.3) vì job mới quyết định quy tắc người gán đã dùng. (3) Split: cảnh scene_311 có 000000010764 ở val, 4 ảnh còn lại ở train; v2 đưa cả cảnh về train (theo đa số) để không rò ảnh cùng cảnh giữa train và val; 000000011122 của cảnh này nằm trong rework (class_swap). (4) Rework: 14 ảnh lỗi nhãn detection hoặc thiếu file chuyển sang rework_queue.csv, giao lại đúng annotator/reviewer theo jobs.csv, không sửa tay nhãn trong v2. (5) Panoptic: các ảnh chỉ lỗi panoptic (000000009483 category_swap, 000000015079, 000000020107, 000000024567 lỗi luật panoptic) giữ trong v2 vì nhãn detection vẫn đúng; báo team panoptic sửa.
