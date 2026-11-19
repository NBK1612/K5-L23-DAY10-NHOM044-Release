# Dùng AI

| # | Ai dùng | Công cụ | Prompt (tóm tắt) | AI sai ở đâu | Nhóm kiểm lại thế nào |
|---|---|---|---|---|---|
| 1 | Lê Ngọc Dũng (N1) | Codex | Dựa trên đề và kế hoạch Nhóm 044, viết kiểm thiếu ảnh và SHA256 ảnh; chỉ đọc delivery, chỉ thư viện chuẩn, trả type/target; giải thích để người học hiểu. | Chưa ghi nhận lỗi ở hai kiểm tra file này; chưa hoàn thiện kiểm box và các phần N2/N3. | AI đã thử dữ liệu tạm: hash đúng/sai, thiếu file, không đổi đầu vào, kết quả lặp lại; chạy seed 44007 đọc 250 mục, báo 2 missing_image và 1 wrong_hash. Chưa chấm F1; người học cần đọc và tự chạy lại. |
| 2 | Lê Ngọc Dũng (N1) | Codex | Thêm kiểm ID lớp và khung tràn, dung sai 0.01 theo checker; giữ ưu tiên lỗi file; chỉ đọc delivery. | Chưa ghi nhận lỗi trong 11 ca thử; duplicate_box và N2/N3 chưa được nối. | AI thử 11 ca dữ liệu tạm: lớp sai, 4 cạnh, dung sai, ưu tiên, giữ nguyên đầu vào. Seed 44007 báo 2 missing_image, 1 wrong_hash, 1 invalid_class, 1 bbox_out_of_bounds. Chưa đo F1; Anh đọc và tự kiểm lại. |
