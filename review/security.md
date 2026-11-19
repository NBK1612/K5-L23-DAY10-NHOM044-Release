# Bảo mật khi làm việc

> Nhóm 044, lô seed 44007 (`lab250_e9b358f6faee`, 250 ảnh COCO). Người viết: Bùi Quang Huy (N2).

## 1. Ca USB: hết giờ, muốn chép lô về nhà làm tiếp
Không chép lô `lab250_e9b358f6faee` ra USB hay máy cá nhân. Lô có ảnh người thật, nhãn và thông tin người gán/kiểm (A0x, Q0x trong `jobs.csv`); ra khỏi máy công ty là mất kiểm soát: không biết ai đọc, không xoá lại được, không có nhật ký truy cập.
Gửi qua email cá nhân cũng là đưa dữ liệu ra ngoài, chỉ là đổi đường đi. Nhờ người khác làm hộ thì người đó không được giao job, không chịu trách nhiệm, và lineage ghi sai người làm.
Cách đúng: báo trưởng nhóm (N4) là chưa xong, ghi rõ đang dừng ở ảnh/bước nào, hôm sau làm tiếp trên máy công ty. Nếu trễ hạn thì xin lùi hạn hoặc chia lại việc, không mang dữ liệu đi.

## 2. Năm việc: được hay không
- **Mở job được giao trên máy công ty: Được.** Đúng người, đúng việc, đúng máy; mọi thao tác nằm trong hệ thống có kiểm soát.
- **Chụp màn hình gửi người quen: Không.** Ảnh chụp chứa ảnh gốc và nhãn, lọt ra ngoài là rò dữ liệu; người nhận không có quyền xem.
- **Chép lô ra USB: Không.** USB dễ mất, không mã hoá, không thu hồi được; lô có ảnh người thật (xem mục 3).
- **Dán nhãn thật lên chatbot: Không.** Dữ liệu đi tới máy chủ bên thứ ba, có thể bị lưu hoặc dùng để huấn luyện. Ở lab này ảnh COCO là công khai nên được hỏi AI và đã ghi vào `prompts.md`; dữ liệu thật của dự án thì không.
- **Đưa tài khoản CVAT cho đồng nghiệp: Không.** Nhật ký sẽ ghi sai người gán/kiểm, phá truy vết (`truyvet.csv`, `notify`); đồng nghiệp cần quyền thì xin cấp tài khoản riêng.

## 3. Ca khuôn mặt: thấy mặt người rõ trong lô, guideline yêu cầu làm mờ
Ví dụ trong lô của nhóm: ảnh `000000001296` (Huy mở xem): cận cảnh một cô gái tóc đen đang cầm điện thoại, khuôn mặt nhìn thẳng, chiếm phần lớn khung hình, đủ rõ để nhận ra danh tính; phía sau còn người khác bị mờ.
Kết luận: **Hold** lô (không Release, không đưa vào train) và **báo người phụ trách dữ liệu/bảo mật** (qua trưởng nhóm N4), kèm tên ảnh và vị trí khuôn mặt.
Không tự làm mờ hay xoá ảnh vì: (1) làm mờ là sửa ảnh, ghi đè lô v1 đã giao, sai hash trong manifest và phá lineage; (2) quyết định ẩn danh hoá theo cách nào, cho ảnh nào, là việc của người phụ trách, người gán nhãn làm là vượt quyền; (3) tự xử lý một ảnh thì các ảnh khác cùng vấn đề không ai biết để rà. Việc làm mờ, nếu được duyệt, sẽ ra một phiên bản mới có lô cha và ghi lineage.
