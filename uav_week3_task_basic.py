import cv2

filenames = ["dots1.png", "dots2.png", "dots3.png"]

# 设置检测规则
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 50
params.maxArea = 5000

params.filterByColor = False
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# 创建检测器，所有图片共用
detector = cv2.SimpleBlobDetector_create(params)

for filename in filenames:
    img = cv2.imread(filename)

    if img is None:
        print("Could not read:", filename)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(gray)

    result = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        (0, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print(filename, "- detected:", len(keypoints))
    cv2.imshow(filename + " - Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()