# Region-Proposal-for-Object-Detection
- Tạo ra các region proposals bằng thuật toán Selective Search
- Phân loại các ROIs
- Áp dụng NMS để loại bỏ bớt các bounding boxes chồng chập.

Cái này giống dùng pyramid và sliding window chỉ khác duy nhất chỗ lấy regions proposals.

Nhận thấy output hay đưa ra những bounding boxes không chứa vật thể nào.
Bài tiếp theo sẽ áp dụng RCNN để xem kết quả cải thiện nhiều hơn không.