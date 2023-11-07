import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
def filter_signs_by_color(image):
    """Lọc các đối tượng màu đỏ và màu xanh dương - Có thể là biển báo.
        Ảnh đầu vào là ảnh màu BGR
    """
    # Chuyển ảnh sang hệ màu HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lọc màu đỏ cho stop và biển báo cấm
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1) # dải màu đỏ thứ nhất
    mask_2 = cv2.inRange(image, lower2, upper2) # dải màu đỏ thứ hai
    mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau

    # Lọc màu xanh cho biển báo điều hướng
    lower3, upper3 = np.array([85, 130, 90]), np.array([150, 255, 255])
    mask_b = cv2.inRange(image, lower3,upper3)

    # Kết hợp các kết quả
    mask_final  = cv2.bitwise_or(mask_r,mask_b)
    return mask_final

def get_boxes_from_mask(mask):
    """Tìm kiếm hộp bao biển báo
    """
    bboxes = []

    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # Lọc các vật quá nhỏ, có thể là nhiễu
        if w < 20 or h < 20:
            continue
        # Lọc các vật quá lớn
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        # Loại bỏ các vật có tỷ lệ dài / rộng quá khác biệt
        if w / h > 2.0 or h / w > 2.0:
            continue
        bboxes.append([x, y, w, h])
    return bboxes

    
cap = cv2.VideoCapture(0)

while (True):
    ret, frame =cap.read()
    ####################################################
    # results = []

    # mask = filter_signs_by_color(frame) # lọc theo màu sắc
    # bboxes = get_boxes_from_mask(mask) # tìm kiếm khung bao của các vật từ mặt nạ màu sắc
    # draw = frame.copy() # Sao chép ảnh màu tương ứng để vẽ lên
    # for bbox in bboxes:
    #     x, y, w, h = bbox
    #     #Vẽ khối hộp bao quanh biển báo      
    #     cv2.rectangle(draw, (x,y), (x+w,y+h), (0,255,255), 2) # vẽ hình chữ nhật bao quanh vật
    #  #results.append(draw)
    # cv2.imshow("Video", draw)
    #########################################################
    # model = cv2.dnn.readNetFromONNX("traffic_sign_classifier_lenet_v2.onnx")

    # # Hàm phát hiện biển báo
    # def detect_traffic_signs(frame, model, draw=None):
 
    #     # Các lớp biển báo
    #     classes = ['unknown', 'left', 'no_left', 'right',
    #            'no_right', 'straight', 'stop']

    #     # Phát hiện biển báo theo màu sắc
    #     mask = filter_signs_by_color(frame)
    #     bboxes = get_boxes_from_mask(mask)

    #     # Tiền xử lý
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame = frame.astype(np.float32)
    #     frame = frame / 255.0

    #     # Phân loại biển báo dùng CNN
    #     signs = []
    #     for bbox in bboxes:
    #         # Cắt vùng cần phân loại
    #         x, y, w, h = bbox
    #         sub_image = frame[y:y+h, x:x+w]

    #         if sub_image.shape[0] < 20 or sub_image.shape[1] < 20:
    #             continue

    #         # Tiền xử lý
    #         sub_image = cv2.resize(sub_image, (32, 32))
    #         sub_image = np.expand_dims(sub_image, axis=0)

    #         # Sử dụng CNN để phân loại biển báo
    #         model.setInput(sub_image)
    #         preds = model.forward()
    #         preds = preds[0]
    #         cls = preds.argmax()
    #         score = preds[cls]

    #         # Loại bỏ các vật không phải biển báo - thuộc lớp unknown
    #         if cls == 0:
    #             continue

    #         # Loại bỏ các vật có độ tin cậy thấp
    #         if score < 0.9:
    #             continue

    #         signs.append([classes[cls], x, y, w, h])

    #         # Vẽ các kết quả
    #         if draw is not None:
    #             text = classes[cls] + ' ' + str(round(score, 2))
    #             cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 1)
    #             cv2.putText(draw, text, (x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    #     return signs

    # draw = frame.copy()
    # signs = detect_traffic_signs(frame, model, draw=draw)
    # cv2.imshow("Video", draw)
    ####################################################################
    from keras.models import load_model
    model = load_model('my_model6.keras')
    # Hàm phát hiện biển báo
    def detect_traffic_signs(frame, model, draw=None):
 
        # Các lớp biển báo
        classes = ['unkown','Speed limit (20km/h)',
            'Speed limit (30km/h)', 
            'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 
           'Speed limit (70km/h)', 
            'Speed limit (80km/h)', 
           'End of speed limit (80km/h)', 
           'Speed limit (100km/h)', 
           'Speed limit (120km/h)', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
           'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
           'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
           'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
           'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons' ]

        # Phát hiện biển báo theo màu sắc
        mask = filter_signs_by_color(frame)
        bboxes = get_boxes_from_mask(mask)

        # Tiền xử lý
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = frame / 255.0

        # Phân loại biển báo dùng CNN
        signs = []
        for bbox in bboxes:
            # Cắt vùng cần phân loại
            x, y, w, h = bbox
            sub_image = frame[y:y+h, x:x+w]

            if sub_image.shape[0] < 20 or sub_image.shape[1] < 20:
                continue

            # Tiền xử lý
            global label_packed
            sub_image = cv2.resize(sub_image, (30, 30))
            sub_image = np.expand_dims(sub_image, axis=0)

            # Sử dụng CNN để phân loại biển báo
            sub_image = np.array(sub_image)
       
            pred = model.predict_on_batch(sub_image)
            classify= np.where(pred == np.amax(pred))[1][0]
            score = np.amax(pred)
            # Loại bỏ các vật không phải biển báo - thuộc lớp unknown
            if classify == 0:
                continue

            # Loại bỏ các vật có độ tin cậy thấp
            if score < 0.95:
                continue

            signs.append([classes[classify], x, y, w, h])

            # Vẽ các kết quả
            if draw is not None:
                text = classes[classify] + ' ' + str(round(score, 2))
                cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 1)
                cv2.putText(draw, text, (x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return signs
    draw = frame.copy()
    signs = detect_traffic_signs(frame, model, draw=draw)
    cv2.imshow("Video", draw)
    #################################################################333333
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()