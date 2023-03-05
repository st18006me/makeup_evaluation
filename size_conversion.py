import numpy as np
import cv2 as cv

float_img = cv.imread('makeup/model.jpg')
ref_img = cv.imread('makeup/makeup.jpg')

akaze = cv.AKAZE_create()
float_kp, float_des = akaze.detectAndCompute(float_img, None)
ref_kp, ref_des = akaze.detectAndCompute(ref_img, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(float_des, ref_des, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# 適切なキーポイントを選択
ref_matched_kpts = np.float32(
    [float_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32(
    [ref_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# ホモグラフィを計算
H, status = cv.findHomography(
    ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)

# 画像を変換
warped_image = cv.warpPerspective(
    float_img, H, (ref_img.shape[1], ref_img.shape[0]))

cv.imwrite('makeup/model_size_conversion.jpg', warped_image)
