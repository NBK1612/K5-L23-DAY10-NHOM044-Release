# Báo cáo nhóm 044

## 1. Số liệu luyện tập
- F1 theo seed (rule / cross_check / overall), số làm tròn do `score_findings.py` in, mỗi danh sách trọn vẹn chạy một lần:

  | Lô | Lỗi luật báo | rule | cross_check | overall | Ghi chú |
  |---|---:|---:|---:|---:|---|
  | 44007 (chỉ lỗi luật, P1) | 12 | ≈ 1.0 | — | ≈ 0.7 | chưa có lỗi gán nhãn |
  | 42 (chỉ lỗi luật, P1) | 11 | ≈ 1.0 | — | ≈ 0.8 | chưa có lỗi gán nhãn |
  | clean, seed 5, 50 ảnh | 0 | ≈ 1.0 | — | ≈ 1.0 | lô không gài lỗi, validator không báo nhầm |
  | 44007 (sau review, P2) | 21 | ≈ 1.0 | ≈ 0.7 | ≈ 0.9 | 12 lỗi luật + 9 lỗi gán nhãn đã xác nhận qua review |

  Lỗi luật seed 44007: 2 missing_image, 1 wrong_hash, 1 invalid_class, 1 bbox_out_of_bounds, 2 duplicate_box, 1 guideline_mismatch, 1 scene_leak, 1 invalid_category_id, 2 thing_overlap. Lô sạch 0 lỗi cho thấy `thing_overlap` (so `area` với số pixel đếm từ mask) không cần dung sai: ở dữ liệu gốc hai số bằng nhau tuyệt đối.
- Số ứng viên (đo bằng `validate.py --candidates`; ảnh đã có lỗi luật không đưa vào hàng đợi). Ngưỡng được chỉnh qua 3 vòng, mỗi vòng review đủ hàng đợi rồi mới chấm một lần:

  | Vòng | Lô sạch (seed 5, 50 ảnh) | Seed 44007 | Lô khác | Lỗi gán nhãn `confirmed` | F1 cross_check |
  |---|---:|---:|---|---:|---:|
  | 1 (ngưỡng chặt) | 0 | 11 ảnh | r42: 13 | 11 | ≈ 0.6 |
  | 2 (nới ngưỡng) | 3 | 26 ứng viên / 24 ảnh | r42: 32 | 13 | ≈ 0.5 |
  | 3 (bbox_drift đo tịnh tiến + tiêu chí xác nhận chặt) | 3 | 28 ứng viên / 27 ảnh | r777 (`--opaque`): 31 | 9 | ≈ 0.7 |

  Vòng 1 bỏ sót nhiều (không có ứng viên `bbox_drift`/`instance_merge`); vòng 2 thêm ứng viên nhưng nhóm xác nhận cả ca lệch tự nhiên; vòng 3 xem lại chéo 13 dòng `confirmed` (`review/xem_lai_cheo.csv`) và áp tiêu chí chặt cho mọi dòng. Lô sạch còn 3 ứng viên, đều `instance_merge` do che khuất.
- Ngưỡng đã chọn và cách chọn (không dùng `_private`):
  - **Cách ghép:** đổi box YOLO sang pixel, ghép lớp qua tên (`classes.txt` ↔ `categories`), mỗi box ghép với segment thing có IoU bbox cao nhất, bỏ segment `iscrowd`. Tín hiệu chính là **tỉ lệ pixel của mask nằm trong box** (đếm từ mask PNG bằng `scripts/pngutil.py`), không chỉ IoU bbox.
  - **Ngưỡng hiện tại (vòng 3, `validator/crosscheck.py`):**

    | Dấu hiệu | Điều kiện báo ứng viên |
    |---|---|
    | `class_swap` / `category_swap` | box và segment trùng vị trí (IoU ≥ 0,30) nhưng khác lớp; segment ≥ 200 px hoặc box ≥ 0,2% ảnh |
    | `bbox_drift` | ghép box với segment **cùng lớp**; hai cạnh đối diện dịch cùng chiều (tịnh tiến) ≥ 8% kích thước vật và ≥ 4 px; segment ≥ 500 px; bỏ qua nếu box trùng một segment khác lớp tốt hơn (để `class_swap` xử lý) |
    | `missing_box` | segment thing không có box nào đạt IoU ≥ 0,20; segment ≥ 600 px và ≥ 0,4% ảnh |
    | `instance_merge` | một segment có ≥ 350 px nằm trong mỗi box của ít nhất 2 box cùng lớp; mỗi box chứa 15–85% segment, tổng ≥ 70%; hai box IoU ≤ 0,40 |

  - **Cách chọn:** chạy trên lô sạch (mọi ứng viên ở đó là lệch tự nhiên của COCO: vật bị che, crowd) và đặt ngưỡng ngoài vùng nhiễu đo được, đồng thời giữ hàng đợi seed 44007 đủ nhỏ để 5 người xem hết. Ví dụ `bbox_drift`: trên lô sạch, độ tịnh tiến lớn nhất của cặp đúng (segment ≥ 200 px) là 0,057, nên chọn 0,08; vật < 100 px dịch tới 0,07–0,15 chỉ vì làm tròn pixel nên phải loại vật nhỏ. Đã thử thêm trên lô `--opaque` seed 777 (tên `img_xxxx`) để chắc ngưỡng không chỉ khớp một lô.
  - **Rủi ro đã biết:** đề ẩn có thể có độ lệch box khác lô luyện tập; `instance_merge` vẫn sinh nhiều ca che khuất tự nhiên (vòng 3: 13/13 ứng viên `instance_merge` của seed 44007 bị `rejected`). `class_swap` và `category_swap` cùng một dấu hiệu, bên nào sai do người xem ảnh quyết định.

## 2. Vì sao báo hết ứng viên làm F1 giảm?
F1 là trung bình điều hoà của precision (báo đúng / tổng số báo) và recall (bắt được / tổng lỗi thật), nên báo thừa bị trừ ngang bỏ sót. Báo hết ứng viên làm recall tăng chút ít nhưng precision giảm mạnh, vì phần lớn ứng viên là lệch tự nhiên giữa detection và panoptic của COCO (vật bị che, đứng sát nhau, crowd), không phải lỗi gài.

Số của nhóm trên seed 44007 (mỗi dòng là một danh sách trọn vẹn sau một vòng review, chấm một lần):

| Danh sách lỗi gán nhãn nộp | Số dòng `confirmed` | F1 cross_check | F1 tổng |
|---|---:|---:|---:|
| Vòng 2: xác nhận rộng (gồm ca lệch tự nhiên, ca mơ hồ bus/truck, cup/wine glass) | 13 | ≈ 0.5 | ≈ 0.8 |
| Vòng 3: cùng hàng đợi, chỉ xác nhận ca chắc chắn qua xem lại chéo | 9 | ≈ 0.7 | ≈ 0.9 |

- Hàng đợi vòng 3 có 28 ứng viên. Nếu nộp cả 28 mà không xem ảnh, ít nhất 13 ứng viên `instance_merge` mà cả nhóm đã mở ảnh và thấy là che khuất tự nhiên sẽ thành báo thừa, nên F1 còn thấp hơn vòng 2.
- Giữa vòng 2 và vòng 3, nhóm bỏ 5 dòng `confirmed` không chắc chắn (xe ở xa bus/truck, chai cắm hoa bottle/vase, cốc nhỏ cup/wine glass, ngựa vằn bị cắt mép, nải chuối bị che) và thêm 1 dòng `bbox_drift` mới; F1 cross_check tăng từ ≈ 0.5 lên ≈ 0.7. Mỗi dòng sai làm tăng FP, và ca chọn sai chiều (`class_swap` thay vì `category_swap`) còn bị tính vừa FP vừa FN.
- Vì vậy nhóm chỉ `confirmed` khi nhìn ảnh thấy chắc chắn, còn lại ghi `rejected` kèm lý do (20/29 dòng review là `rejected`).

## 3. Quyết định và lineage
- **QC v1:** 21/250 = 8,4% (12 lỗi luật + 9 lỗi gán nhãn đã xác nhận qua review bằng mắt).
- **Kết luận:** **HOLD** — tỷ lệ lỗi 8,4% vượt xa ngưỡng cho phép 2% theo tiêu chuẩn QC. Lô v1 không đạt chuẩn bàn giao huấn luyện.
- **Lineage:** Đã ghi sự kiện `decision: HOLD` cho lô `lab250_e9b358f6faee` và sau đó phát hành lô cha-con `release: lab250_e9b358f6faee_v2` với parent là `lab250_e9b358f6faee`.
- **Vì sao training run R-021 phải xem lại hoặc train lại:**
  Truy vấn `scripts/lineage.py impact --release-id lab250_e9b358f6faee` trả về: `R-021: affected=True action=review_or_retrain`.
  Lý do là mô hình `R-021` đã được huấn luyện trực tiếp trên tập dữ liệu v1 chứa tới 8,4% lỗi gài (gồm ảnh thiếu file, nhầm nhãn, box lệch, box trùng...). Các trọng số (weights) của mô hình đã học theo các tín hiệu sai này. Việc sau đó nhóm tạo ra lô `v2` sạch hơn và loại bỏ ảnh lỗi hoàn toàn **không tự động làm sạch trọng số của model đã train**. Do đó, `R-021` bắt buộc phải được đánh giá lại mức độ suy giảm chất lượng hoặc phải train lại từ đầu trên lô dữ liệu v2.

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
- **Đề nghị khi chấm:** `export_submission.py --import` tạo lại lô v1 trên máy người chấm. Nếu máy đó là Windows với Python ≥ 3.14, `release_id` sẽ khác `lab250_e9b358f6faee`, `check_handover.py` tìm `handover/<mã khác>_v2` và báo FAIL H4 rồi dừng, dù bài đúng. Nhóm đã giả lập (đổi mức nén PNG, pixel giữ nguyên) và thấy đúng hiện tượng này. Đề nghị chạy bằng Python ≤ 3.13.

**Ký tự xuống dòng làm lệch sha256 khi clone trên Windows.**
- Git trên Windows thường bật `core.autocrlf=true`: khi checkout, nhãn `.txt` trong `submission/r44007/handover/` bị đổi LF → CRLF, sha256 khác `manifest.csv` nên `check_handover.py` báo FAIL H7, H8, H15 (đã gặp trên máy N4), dù bytes đã commit đúng (kiểm từ bytes trong git: 17/17 PASS).
- Nhóm xử lý: thêm `.gitattributes` với `submission/**/handover/** -text` để git giữ nguyên byte; checkout lại thì 17/17 PASS trên Windows.
