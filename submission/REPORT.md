# Báo cáo nhóm 044

## 1. Số liệu luyện tập
- F1 theo seed (rule / cross_check / overall), số làm tròn do `score_findings.py` in, mỗi danh sách trọn vẹn chạy một lần:

  | Lô | Lỗi luật báo | rule | cross_check | overall | Ghi chú |
  |---|---:|---:|---:|---:|---|
  | 44007 (chỉ lỗi luật, P1) | 12 | ≈ 1.0 | — | ≈ 0.7 | chưa có lỗi gán nhãn |
  | 42 (chỉ lỗi luật, P1) | 11 | ≈ 1.0 | — | ≈ 0.8 | chưa có lỗi gán nhãn |
  | clean, seed 5, 50 ảnh | 0 | ≈ 1.0 | — | ≈ 1.0 | lô không gài lỗi, validator không báo nhầm |
  | 44007 (sau review, P2) | <<>> | <<>> | <<>> | <<>> | <<>> |

  Lỗi luật seed 44007: 2 missing_image, 1 wrong_hash, 1 invalid_class, 1 bbox_out_of_bounds, 2 duplicate_box, 1 guideline_mismatch, 1 scene_leak, 1 invalid_category_id, 2 thing_overlap. Lô sạch 0 lỗi cho thấy `thing_overlap` (so `area` với số pixel đếm từ mask) không cần dung sai: ở dữ liệu gốc hai số bằng nhau tuyệt đối.
- Số ứng viên trên lô sạch (`--no-faults`, seed 5, 50 ảnh): **0**; hàng đợi trên seed luyện tập 44007: **11 ảnh** (seed 42: 13 ảnh). Số do N3 đo bằng `validate.py --candidates` (ghi trong phần đầu `validator/crosscheck.py`). Mỗi ảnh giữ một ứng viên; ảnh đã có lỗi luật không đưa vào hàng đợi.
- Ngưỡng đã chọn và cách chọn (không dùng `_private`):
  - **Cách ghép:** đổi box YOLO sang pixel, ghép lớp qua tên (`classes.txt` ↔ `categories`), mỗi box ghép với segment thing có IoU bbox cao nhất, bỏ segment `iscrowd`. Tín hiệu chính là **tỉ lệ pixel của mask nằm trong box** (đếm từ mask PNG bằng `scripts/pngutil.py`), không chỉ IoU bbox.
  - **Ngưỡng:**

    | Dấu hiệu | Điều kiện báo ứng viên |
    |---|---|
    | `class_swap` / `category_swap` | box và segment trùng vị trí (IoU ≥ 0,35) nhưng khác lớp; segment ≥ 400 px hoặc box ≥ 0,3% ảnh |
    | `bbox_drift` | cùng lớp, IoU ≥ 0,05, tỉ lệ pixel mask trong box < 0,70, segment ≥ 600 px |
    | `missing_box` | segment thing không có box nào đạt IoU ≥ 0,20, segment ≥ 1200 px và ≥ 0,8% ảnh |
    | `instance_merge` | một segment có ≥ 250 px nằm trong mỗi box của ít nhất 2 box cùng lớp, hai box IoU < 0,5 |

  - **Cách chọn:** chạy trên lô sạch (mọi ứng viên ở đó là lệch tự nhiên của COCO: vật bị che, crowd) và nâng ngưỡng đến khi lô sạch còn 0 ứng viên, đồng thời kiểm hàng đợi seed 44007 đủ nhỏ để 5 người xem hết. Ví dụ `bbox_drift`: trên lô sạch, tỉ lệ pixel trong box của cặp đúng thấp nhất là 0,728, nên chọn ngưỡng 0,70 nằm dưới mức đó.
  - **Rủi ro đã biết:** ngưỡng `bbox_drift` sát mức lệch tự nhiên (0,70 so với 0,728); đề ẩn có thể có độ lệch box khác, nên ngưỡng chặt có thể bỏ sót. `class_swap` và `category_swap` cùng một dấu hiệu, bên nào sai do người xem ảnh quyết định.

## 2. Vì sao báo hết ứng viên làm F1 giảm?
<<Trả lời bằng số của nhóm.>>

## 3. Quyết định và lineage
<<QC v1 a/b, kết luận, vì sao R-021 phải xem lại hoặc train lại.>>

## 4. Hash chứng minh gì, không chứng minh gì?
**Chứng minh:**
- File không bị đổi dù một byte kể từ lúc lập manifest. `manifest.csv` của v2 ghi sha256 từng ảnh và nhãn; `.sha256` của gói zip cho phép người nhận kiểm cả lô một lần.
- v1 không bị ghi đè: `release_id` = `lab250_` + 12 ký tự đầu sha256 của `manifest.json` + `metadata.json`; sửa v1 thì mã đổi (`check_handover.py` H1). Nhãn trong v2 có cùng sha256 với nhãn v1 nên không bị sửa tay (H7).
- Phát hiện file hỏng hoặc bị tráo: lỗi `wrong_hash` là sha256 tự tính từ bytes khác với manifest.

**Không chứng minh:**
- **Nhãn đúng.** Một nhãn sai (`class_swap`, `bbox_drift`...) vẫn có hash hợp lệ; hash chỉ nói file không đổi, không nói nội dung đúng. Đúng/sai phải đối chiếu hai nguồn và mở ảnh xem.
- **Ai làm, làm lúc nào, có được phép không.** Hash không ký tên; người có quyền ghi vẫn sửa file rồi lập lại manifest được. Nguồn gốc phải dựa vào lineage, `truyvet.csv` và quy trình.
- **Ảnh còn dùng được.** Khi `wrong_hash`, chỉ biết bytes khác manifest; phải mở ảnh mới biết ảnh hỏng (đưa vào rework) hay chỉ manifest ghi sai (ghi hash mới trong v2).
- Hash cũng nhạy với cách mã hoá: cùng pixel nhưng nén PNG khác thì hash khác (xem §6), nên hash so bytes, không so nội dung ảnh.

## 5. Truy vết
**Tiêu chí** (đọc `incoming/ticket.md` để biết các cảnh team AI báo, `guideline/changelog.md` để biết lớp bị đổi quy tắc từ v1.2 sang v1.3):
- **`tier_a`:** ảnh nằm trong các cảnh team AI nêu, thuộc job có guideline **cũ** (theo `jobs.csv`), và có ít nhất một **box detection** (trong `labels/` của lô) thuộc lớp bị đổi quy tắc.
- **`tier_b`:** các ảnh khác cùng điều kiện (job guideline cũ, có box lớp đó) nhưng nằm ngoài các cảnh đã được báo: cùng rủi ro, chưa ai phát hiện.
- **`notify`:** mã người gán và người kiểm (`jobs.csv`) của các job chứa ảnh tầng a và b.

**Vì sao lấy guideline theo job, không theo từng ảnh:** guideline được giao ở đầu job, người gán làm cả 25 ảnh của job theo phiên bản đó, nên phiên bản ghi ở job mới phản ánh quy tắc thật đã dùng. Guideline ghi ở từng ảnh trong `metadata.json` có thể chính là chỗ sai (lỗi `guideline_mismatch`); dùng nó để lọc thì sẽ bỏ sót hoặc kéo nhầm ảnh.

**Tầng a và b khác nhau:** tầng a là ảnh đã có triệu chứng được báo, cần mở lại trước; tầng b là ảnh chưa ai báo nhưng gán theo cùng quy tắc cũ, cần rà chủ động để tránh lỗi tương tự lọt vào lần train sau.

## 6. Bug hoặc điểm chưa rõ trong lab (nếu có)
**Cùng seed nhưng `release_id` khác nhau tuỳ phiên bản Python** (Huy, N2 phát hiện).
- Seed 44007 với Python 3.12 (zlib 1.3.1) ra `lab250_e9b358f6faee`; với Python 3.14/3.15 trên Windows (zlib-ng) ra `lab250_050cd73ecce9`.
- So hai lô: ảnh, nhãn, `metadata.json` và danh sách lỗi gài giống hệt; chỉ 241/250 mask PNG khác sha256. Giải nén thử 40 mask thì pixel giống hệt: `pngutil.write_rgb` nén bằng `zlib`, zlib-ng nén ra byte khác, nên `mask_sha256` trong `manifest.json` và `release_id` đổi.
- Hệ quả: các máy trong nhóm không cùng `release_id`; nếu dựng v2 ở máy khác phiên bản thì hash mask lệch.
- Nhóm xử lý: cả nhóm chạy mọi script bằng `py -3.12`. Không sửa `scripts/`.
