import numpy as np
from PIL import Image
import sys

# 画像の読み込み
image1 = Image.open("makeup/sample.jpg")
image2 = Image.open("makeup/model_size_conversion.jpg")

# RGB画像に変換
image1 = image1.convert("RGB")
image2 = image2.convert("RGB")

# NumPy配列へ変換
im1_u8 = np.array(image1)
im2_u8 = np.array(image2)

# サイズや色数が違うならエラー
if im1_u8.shape != im2_u8.shape:
    print("サイズが違います")
    sys.exit()

# 負の値も扱えるようにnp.int16に変換
im1_i16 = im1_u8.astype(np.int16)
im2_i16 = im2_u8.astype(np.int16)

# 差分配列作成
diff_i16 = im2_i16 - im1_i16

# np.uint8型で扱える値に変換
diff_n_i16 = ((diff_i16 + 256) // 2)

# NumPy配列をnp.uint8型に変換
diff_u8 = diff_n_i16.astype(np.uint8)

# PIL画像に変換
diff_img = Image.fromarray(diff_u8)

diff_img.save("makeup/model_difference.jpg")
# 画像表示
diff_img.show()
