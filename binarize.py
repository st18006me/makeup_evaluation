import cv2
import numpy as np

# 赤色の検出
def detect_red_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([120, 30, 50])
    upper_color = np.array([180, 255, 255])
    mask_tehon = cv2.inRange(hsv, lower_color, upper_color)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask_tehon)

    return mask_tehon, masked_img

  
img2 = cv2.imread("makeup/model_difference.jpg")
img1 = cv2.imread("makeup/sample_difference.jpg")

# 色検出（赤、緑、青）
red_mask, red_masked_img = detect_red_color_sabun(img1)
sabun_mask, sabun_masked_img = detect_red_color_sabun(img2)

whole_area1=red_mask.size
white_area1=cv2.countNonZero(red_mask)

whole_area2=sabun_mask.size
white_area2=cv2.countNonZero(sabun_mask)

# 結果を出力
print('一致率:'+str(100-white_area2/white_area1*100)+'%')
cv2.imwrite("makeup/red_mask_model.png", red_masked)
cv2.imwrite("makeup/red_mask_sample.png", sabun_mask)

#画像の表示
cv2.imshow("OpenCV_sample",red_mask)
cv2.imshow("OpenCV_makeup",sabun_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
