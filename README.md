# BiBo-Kit---Panorama
OJT 2015

Image stitching prototype.

Project Startdate: 13.01.2015

#HOW TO USE:

g++ -std=c++0x -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"Stitcher.d" -MT"Stitcher.d" -o "Stitcher.o" "Stitcher.cpp" && g++ -std=c++0x -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"main.d" -MT"main.d" -o "main.o" "main.cpp" && g++ -fopenmp -o "Stitch"  Stitcher.o main.o   -lexiv2 -lboost_system -lboost_filesystem -lopencv_core -lopencv_calib3d -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_stitching

#CHANGELOG:

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
