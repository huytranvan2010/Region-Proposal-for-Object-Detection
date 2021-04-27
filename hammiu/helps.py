import cv2

def selective_search(image, method="fast"):
    # khởi tạo OpenCV's selective search impleamentation vaf set image đầu vào
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    # kiêm tra chế độ của Selective search là "fast" hay "quality"
    if method == "fast":     # nhanh, ít chính xác
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()     # chậm, chính xác hơn

    rects = ss.process()    #chạy thuật toán

    return rects    # trả về các region proposals (rectangles) list of (x, y, w, h)



