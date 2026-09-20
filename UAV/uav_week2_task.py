import cv2 
import numpy as np

filename = "test.JPG"
img = cv2.imread(filename)

if img is None:
    raise SystemExit("Cannot read the image. Please check the file path.")
print("Picture's shape:", img.shape)
print("Height:", img.shape[0], "Width:", img.shape[1],
      "Number of channels (B, G, R):", img.shape[2])

blue_only = img.copy()
blue_only[:,:,1] = 0  
blue_only[:,:,2] = 0

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
brighter = gray.astype(np.float32) + 40
brighter = np.clip(brighter, 0, 255).astype(np.uint8)

bluer = img.astype(np.float32)
bluer[:,:,0] += 30
bluer = np.clip(bluer,0,255).astype(np.uint8)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color_ranges = [
    ("Black",   [0, 0, 0],      [179, 255, 49]),
    ("Gray",    [0, 0, 50],     [179, 59, 199]),
    ("White",   [0, 0, 200],    [179, 59, 255]),
    ("Red",     [0, 60, 50],    [10, 255, 255]),
    ("Orange",  [11, 60, 50],   [24, 255, 255]),
    ("Yellow",  [25, 60, 50],   [34, 255, 255]),
    ("Green",   [35, 60, 50],   [85, 255, 255]),
    ("Cyan",    [86, 60, 50],   [99, 255, 255]),
    ("Blue",    [100, 60, 50],  [129, 255, 255]),
    ("Purple",  [130, 60, 50],  [150, 255, 255]),
    ("Magenta", [151, 60, 50],  [169, 255, 255]),
]


cv2.imshow("Original", img)
cv2.imshow("blue_only", blue_only)
cv2.imshow("gray", gray)
cv2.imshow("brighter gray", brighter)
cv2.imshow("bluer",bluer)

for name, lower, upper in color_ranges:
    
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    if name == "Red":
        lower_red2 = np.array([170, 60, 50], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask, red_mask2)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow(name, result)

cv2.waitKey(0)
cv2.destroyAllWindows()