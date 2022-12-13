import cv2 as cv
import numpy as np
import dlib


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

# convert to gray img


def swap_face(act_img1, act_img2, predictor):
    act_img1_gray = cv.cvtColor(act_img1, cv.COLOR_BGR2GRAY)
    act_img2_gray = cv.cvtColor(act_img2, cv.COLOR_BGR2GRAY)
    # Tạo 1 bảng màu đen giống với img1
    mask = np.zeros_like(act_img1_gray)
    mask2 = np.zeros_like(act_img2_gray)
    img2_new_face = np.zeros_like(act_img2)
    # Hàm dlib sử dụng để nhận diện khuôn mặt : HOG and Linear SVM
    detector = dlib.get_frontal_face_detector()
    # hàm giúp lấy 68 điểm trên khuôn mặt
    faces_1 = detector(act_img1_gray)
    for face in faces_1:
        # create 68 point
        landmarks = predictor(act_img1_gray, face)
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        # Mảng chứa cặp điểm (x,y)
        points = np.array(landmarks_points, np.int32)
        # cover face use func cv.convexhull
        # convexHull là một đường cong lồi xung quanh một vật thể.
        # Một đường cong lồi luôn phình ra, hoặc ít nhất là phẳng.
        convexhull = cv.convexHull(points)
        # vẽ một đa giác lồi đầy. tra ve mask
        cv.fillConvexPoly(mask, convexhull, 255)
        # tuong tu voi img_1 voi mask = mask
        face_img_1 = cv.bitwise_and(act_img1, act_img1, mask=mask)
        # Chuyen doi qua hinh chu nhat
        rect = cv.boundingRect(convexhull)
        # take coor face
        (x, y, w, h) = rect
        # func subdiv2d chia nhỏ mặt phẳng thành các tam giác bằng thuật toán Delaunay
        subdiv = cv.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        # lấy điểm trong tam giác
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, np.int32)
        indexes_triangles1 = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles1.append(triangle)
    # tương tự khuôn mặt 2
    faces_2 = detector(act_img2_gray)
    for face in faces_2:

        # Tạo 68 điểm
        landmarks2 = predictor(act_img2_gray, face)
        landmarks_points2 = []
        for n in range(68):
            x = landmarks2.part(n).x
            y = landmarks2.part(n).y
            landmarks_points2.append((x, y))
        points_2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv.convexHull(points_2)
    for triangle_index in indexes_triangles1:
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        # Bao boc tam giac moi
        rect1 = cv.boundingRect(triangle1)
        (x, y, w, h) = rect1
        # crop triangle
        cropped_triangle = act_img1[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        # ánh xạ coor img 1 đến crop_rec
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv.fillConvexPoly(cropped_tr1_mask, points, 255)
        cropped_triangle = cv.bitwise_and(cropped_triangle, cropped_triangle,
                                          mask=cropped_tr1_mask)
        # Triangulation of second face

        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_triangle2 = act_img2[y: y + h, x: x + w]
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
        cv.fillConvexPoly(cropped_tr2_mask, points2, 255)
        cropped_triangle2 = cv.bitwise_and(cropped_triangle2, cropped_triangle2,
                                           mask=cropped_tr2_mask)

        # swap triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        # Kích thước và hướng của tam giác được xác định bởi 3 điểm thay đổi.
        # Được trang bị cả hai bộ điểm, chúng tôi tính toán Biến đổi Affine bằng cách sử dụng hàm
        M = cv.getAffineTransform(points, points2)
        # biến đổi tam giác tỷ lệ 1 ->2
        warped_triangle = cv.warpAffine(cropped_triangle, M, (w, h))

        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area = cv.bitwise_or(
            img2_new_face_rect_area, warped_triangle, mask=None)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    img2_new_face_gray = cv.cvtColor(img2_new_face, cv.COLOR_BGR2GRAY)
    _, background = cv.threshold(
        img2_new_face_gray, 1, 255, cv.THRESH_BINARY_INV)
    # toán tử bitwise_and thực hiện kết hợp bit khôn ngoan của hai mảng tương ứng với hai hình ảnh trong OpenCV,
    # Toán tử bitwise_and trả về một mảng tương ứng với hình ảnh kết quả từ việc hợp nhất hai hình ảnh đã cho cung kich thuoc
    background = cv.bitwise_and(act_img2, act_img2, mask=background)

#  AND theo bit là đúng khi và chỉ khi cả hai pixel đều lớn hơn 0.
# OR theo bit là đúng nếu một trong hai pixel lớn hơn 0.
# XOR bitwise là đúng khi và chỉ khi một trong hai pixel lớn hơn 0, nhưng không phải cả hai.
    result = cv.bitwise_or(background, img2_new_face)

    img2_face_mask = np.zeros_like(act_img2_gray)
    img2_head_mask = cv.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv.bitwise_not(img2_head_mask)

    (x, y, w, h) = cv.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    dst = result
    # Tao mang 3,3 toan so 1
    # 2 dòng dưới làm mờ ảnh
    kernel = np.ones((3, 3), np.float32)/9
    # chúng ta có thể tích hợp một hình ảnh với kernel (thường là ma trận 2d) để áp dụng bộ lọc trên hình ảnh.
    dst = cv.filter2D(dst, -1, kernel)
    # Hàm nhân bản 
    # Ghép ảnh 1 vào ảnh 2 sao cho hòa hợp
    # dst: Hình ảnh nguồn sẽ được sao chép vào hình ảnh đích. 
    #img2_head_mask : Hình ảnh đích mà hình ảnh nguồn sẽ được sao chép vào
    #center_face2: Vị trí tâm của ảnh nguồn trong ảnh đích.
    # cv.NORMAL_CLONE : có 2 cách nhân bản là cv.NORMAL_CLONE và Mixed_Cloning
    # NORMAL_CLONE kết cấu của hình ảnh nguồn được giữ nguyên trong vùng nhân bản. 
    # TMixed_Cloning kết cấu của vùng nhân bản được xác định bởi sự kết hợp của hình ảnh nguồn và đích.
    img = cv.seamlessClone(dst, act_img2, img2_head_mask,
                           center_face2, cv.NORMAL_CLONE)
    #    medianBlur: Hàm hỗ trợ làm mờ, làm mịn ảnh
    # Số 1 là kích thước của sổ trượt
    # Median là giá trị của phần tử nằm giữa trong mảng đã sắp xếp. Ví dụ:
    img = cv.medianBlur(img, 1)
    return img
