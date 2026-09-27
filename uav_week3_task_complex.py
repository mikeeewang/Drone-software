"""
I found that while the code from the "Week 2 Basic Task" handled pixel-art-style images well, 

it struggled to accurately select regions in complex real-world images, such as those featuring carpets. 

My original plan was to manually select the target areas and define specific color parameters to improve processing results; 

however, due to time constraints, I only managed to complete the image selection portion of the code.

I am also very interested in YOLO, so I decided to wrap up this task at this stage.
"""

import cv2
import numpy as np

filenames = ["dots1.png", "dots2.png", "dots3.png"]

def select_detection_area(img):
    points = []
    preview = img.copy

    def on_click(event, x, y, flags, userdata):
        """鼠标点击->触发事件 (opencv 调用这个函数)"""

        if event == cv2.EVENT_LBUTTONDOWN and len(points)<4:
            points += [(x,y)]

            #在预览图片上点击处plot红点
            #cv2.circle(要画的图片, 圆心, 半径, 颜色，线条粗细（-1填满整个dot）)
            cv2.circle(preview, (x,y), 6, (0,0,255), -1)
            cv2.imshow("Click the four corners of the area for detection", preview)

    # 创建窗口，并让它监听鼠标点击
    cv2.namedWindow("Click the four corners of the area for detection")
    cv2.setMouseCallback("Click the four corners of the area for detection", on_click)
    cv2.imshow("Click the four corners of the area for detection", preview)

    #等待user 点击四个角落
    while len(points) < 4:
        # waitKey(20) 每隔约 20 毫秒检查一次键盘输入
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow("Click the four corners of the area for detection")
            return None
        
    # 四个角都点完后，等待按任意键
    cv2.waitKey(0)
    cv2.destroyWindow("Click the four corners of the area for detection")

    # 创建一张与原图同样宽高的全黑图片
    # img.shape[height, weight, color], [:2] 切掉color
    mask = np.zeros(img.shape[:2], dtype = np.unit8)
   
    # 将四个点击位置围出的区域填成白色
    cv2.fillPoly(mask, [np.array(points, dtype = np.int32)], 255)

    return mask

"""逐张处理图片"""

for filename in filenames:

    img = cv2.imread(filename)

    if img is None:
        print("Couldn't read:", filename)
        continue # 跳过这张图片，继续处理下一张

    # 在副本上画检测结果，保留原图供比较
    result = img.copy()

    if filename == "dots1.png":
        # ====================================================
        # dots1：简单图片，使用 Blob 检测方法
        # ====================================================
                
        # Blob 检测器处理灰度图：每个像素只有一个亮度值
        gray = cv2.cvtColor(filename, cv2.COLOR_BGR2GRAY)

        #创建检测参数
        params = cv2.SimpleBlobDetector_Params()

        # 启用面积过滤：只保留指定面积范围内的区域
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000

        # 暂时关闭其他过滤条件
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # 根据上面的参数创建检测器，并在灰度图上检测
        detector = cv2.SimpleBlobDetector_creat(params)
        keypoints = detector.detect(gray)


        # 在原图上画绿色标记
        # DRAW_RICH_KEYPOINTS 会让圆圈大小反映检测到的 Blob 大小

        result = cv2.drawKeypoints(img, keypoints, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        count = len(keypoints)

    else:
        # ====================================================
        # 复杂照片的专用处理
        # ====================================================
        print("Click the 4 corners in the img, then press a key.")
        area_mask = select_detection_area(img)

        if area_mask is None:
            print("Skipped:", filename)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray,0,205)

        b, g, r = cv2.split(img)

        