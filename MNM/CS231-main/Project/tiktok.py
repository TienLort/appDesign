import cv2
# OpenCV-Python là một thư viện mà các nhà phát triển sử dụng để xử lý hình ảnh cho các ứng dụng thị giác máy tính.
# Thư viện này cung cấp nhiều hàm cho các tác vụ xử lý hình ảnh như đọc và ghi hình ảnh cùng lúc,
# xây dựng môi trường 3D từ môi trường 2D cũng như chụp và phân tích hình ảnh từ video.
import numpy as np
import dlib
# Dlib là thư viện mã nguồn mở về Machine Learning
from math import hypot

# Tìm face landmark
# Hàm tìm khuôn mặt có sẵn của thư viện dlib
predictor = dlib.shape_predictor(
    "./__pycache__/shape_predictor_68_face_landmarks.dat")
nose_image = cv2.imread("static/pig_nose.png")

# Hàm xử lý gắn mũi


def attach_nose(frame, nose_image, predictor):
    # Chuyển ảnh mũi về dạng màu xám, Ảnh màu thực chất chỉ là tập hợp của những ma trận số có cùng kích thước
    # sẽ dễ dàng hơn nếu ta chỉ xử lý dữ liệu trên một ma trận số thay vì nhiều ma trận số
    gray = cv2.cvtColor(nose_image, cv2.COLOR_BGR2GRAY)
    # Chuyển ảnh về thành ảnh đen trắng . Ngưỡng để xác định đen và trắng được truyền qua tham số thres.
    # Ảnh đen trắng thường được ứng dụng trong bài toán phân vùng ảnh
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    # Chuyển các giá trị có ngưỡng = 255 về 0
    nose_image[thresh == 255] = 0
    # kernel: Là phần tử cấu trúc.khởi tạo và sử dụng bằng cách sử dụng hàm getStructuringElement().
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nose_image = cv2.erode(nose_image, kernel, iterations=1)
    # Tra ve cau truc hinh dang cua mang
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    # Hàm nhận diện khuôn mặt do dlib cung cấp
    detector = dlib.get_frontal_face_detector()

    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)
        # So do cua mui duoc tra ve
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        # Tinh do dai tuong doi cua mui
        # trả về sqrt(x*x + y*y).
        nose_width = int(hypot(left_nose[0] - right_nose[0],
                               left_nose[1] - right_nose[1]) * 2.2)
        nose_height = int(nose_width * 0.77)
        nose_width = int(abs(left_nose[0] - right_nose[0])*2.2)
        # Tinh vi tri tra ve cua mui
        top_left = (int(center_nose[0] - nose_width / 2),
                    int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                        int(center_nose[1] + nose_height / 2))
        # Thay doi kich thuoc cua anh theo mui
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(
            nose_pig_gray, 100, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width]
        #   Tính toán OR theo bit của hai mảng theo từng phần tử.
        nose_area_no_nose = cv2.bitwise_or(
            nose_area, nose_area, mask=nose_mask)
            
        # Gắn mũi mới vào mặt
        final_nose = cv2.add(nose_area_no_nose, nose_pig)
        frame[top_left[1]: top_left[1] + nose_height,
              top_left[0]: top_left[0] + nose_width] = final_nose

    return frame
