# BiBo-Kit---Panorama
OJT 2015

Image stitching prototype.

Project Startdate: 13.01.2015

#HOW TO USE:

Trong Terminal trỏ đến thư mục ImageStitching, gõ lệnh make

#TEST CASE & RESULT:

Test case: https://drive.google.com/file/d/0B4hX31GyxRr9ejI5WG1Ud1RlYm8/view?usp=sharing

933: http://i57.tinypic.com/e63h2t.jpg
1011: http://i58.tinypic.com/149mryx.jpg

#CHANGELOG:

21/4/2015: v1.1RC1
- Tự động quét thư mục tìm ảnh khi pairwise.txt không tồn tại

20/4/2015:
- Tối ưu quản lý bộ nhớ
- Thay đổi kiểu nhập matching mask

13/04/2015:
- Giảm bộ nhớ sử dụng
- Xử lý bộ ảnh đầu vào không đồng đều

12/04/2015:
- Cải thiện hiệu suất

11/04/2015:
- Thay cách lựa chọn kết quả trong trường hợp nối không đủ
- Đổi cấu trúc trả về tình trạng xử lý
- Giảm chất lượng và dung lượng ảnh preview
- Loại bỏ chế độ nối ảnh FAST và PREVIEW
- Cải thiện hiệu suất

10/04/2015:
- Sửa lỗi khi nối ảnh lần 2

08/04/2015:
- Thêm makefile
- Tự ghi log ra màn hình nếu có lỗi khi ghi vào file log

01/04/2015:
- Cải thiện nhỏ hiệu suất.

31/03/2015:
- Không resize với ảnh có kích thước < 1.25*1080p.
- Sửa lại log.
- Bổ sung 1 số thủ thuật giảm tỉ lệ ghép thất bại, bù lại tăng thời gian xử lí.

30/03/2015: 
- Tăng mức độ ổn định khi ghép với bộ ảnh ko lý tưởng.
- Log chi tiết hơn.
- 2 mép ảnh kết quả có thể liền mạch với nhau tạo thành vòng kín.

1.0: Initial release.
