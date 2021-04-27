# Cách dùng 
# python region_proposal_detection.py --image ./images/cat.jpg

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
from hammiu import selective_search
import numpy as np 
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-m", "--method", type=str, default="fast", choices=["fast", "quality"], help="selective search method")
ap.add_argument("-c", "--conf", type=float, default=0.9, help="minimum probability to consider detection")
ap.add_argument("-f", "--filter", type=str, default=None, help="comma separated list of ImageNet labels to filter on")  # các class được cách nhau bằng dấu ","
args = vars(ap.parse_args())

# lấy label filter
labelFilters = args["filter"]

# Nếu label fiter không rỗng thì chuyển nó về list
if labelFilters is not None:
    labelFilters = labelFilters.lower().split(",")

# load pre-trained ResNet (với weights pre-trained on ImageNet)
print("[INFO] loading ResNet...")
model = ResNet50(weights="imagenet")

# load image
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# Thực hiện selective search lên ảnh ban đầu
print("[INFO] performing selective search with '{}' method".format(args["method"]))
rects = selective_search(image, method=args["method"])
print("[INFO] regions found by selective search".format(len(rects)))

# Khởi tạo 2 list để chứa các region proposals đủ lớn và bounding boxes của chúng
proposals = []
boxes = []

# Duyệt qua các regions proposals
for (x, y, w, h) in rects:
    # Nếu width hay height của region proposal < 10% width hay height của ảnh
    # ban đầu thì bỏ qua (loại bỏ những "small objetc" thường nhận thành False-Positive)
    # Thực sự cái này còn tùy vào ảnh, để chính xác thì ko làm gì cả, máy yêu để có thể tràn Ram
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue    # tiếp tục duyệt đến region proposal khác

    # thỏa mãn sẽ trích xuất roi ra 
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)      # chuyển về RGB color space
    roi = cv2.resize(roi, (224, 224))   # resize cho phù hợp ResNet

    # một số tiền sử lý khác
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # update our proposals và bounding boxes tương ứng
    proposals.append(roi)
    boxes.append((x, y, w, h))

# Sau khi lấy được region proposals đủ lớn và bounding boxes của nó => thực hiện classify cho mỗi ROI
# convert proposal list về numpy array
proposals = np.array(proposals)
print("Proposal shape: ", proposals.shape)
print("Sufficient large region proposals: ", proposals.shape[0])

# classify and decode the predictions
print("[INFO] classifying proposals...")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)     # list of list of tuple (classID, label, prob)

# tạo dictionary để map class label (key) đến list of bounding boxes + probabilities liên quan đến nó
# Nên nhớ ở đây dùng ResNet phân loại nên mỗi region proposal sẽ cho to 1 class (lấy top=1)
labels = {}

# duyệt qua các predictions
for (i, p) in enumerate(preds):     # lấy cả chỉ số i để tí lấy bounding boxes tương ứng
    # lấy thông tin dự đoán cho mỗi region proposal
    (imagenetID, label, prob) = p[0]    # do lấy top=1

    # nếu labelFilter không rỗng (có truyền vào các nhãn cách nhau bằng dấu ',') và label dự đoán
    # được từ ImageNet không có trong danh sách đó thì bỏ qua 
    # Nếu mình truyền vào labelFilter có nghĩa mình chỉ quan tâm những label đó thôi
    if labelFilters is not None and label not in labelFilters:
        continue

    # Lọc các detections có probability lớn hơn threshold
    if prob >= args["conf"]:
        # lấy bounding boxes tương ứng
        (x, y, w, h) = boxes[i]
        box = (x, y, x + w, y + h)      # lưu tọa độ góc trên bên trái và tọa độ góc dưới bên phải

        # nếu chưa có key label thì tạo list rỗng, có rồi thì trả về list đã chữa sẵn giá trị
        L = labels.get(label, [])
        L.append((box, prob))   # chèn values vào
        labels[label] = L   # cập nhật lại values cho key đó

# Áp dụng non-max suppression, có biểu diễn trước khi áp dụng và sau khi áp dụng
# Duyệt qua các label
for label in labels.keys():
    # Ở trong này sẽ lần lượt duyệt qua các label
    print("[INFO] showing results for '{}'".format(label))
    # clone ảnh để vẽ
    clone = image.copy()

    # Duyệt qua các bounding boxes và probs cho label before NMS
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Before NMS", clone)

    # clone ảnh để vẽ sau NMS
    clone = image.copy()

    # Trích xuất bounding boxes và probabilities riêng ra trước khi áp dùng NMS
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])

    boxes = non_max_suppression(boxes, proba)   # xem thêm imutils

    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    
    cv2.imshow("After NMS", clone)
    cv2.waitKey(0)

"""
Lưu ý:
    - Box truyền vào NMS ở dạng (x, y, x + w, y + h)
    - Box lấy ra từ selective search ở dạng (x, y, w, h)
"""


