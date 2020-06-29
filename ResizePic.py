# %%
import cv2

src = cv2.imread('F:/rangduju/cnn-classification/data/0000.png', cv2.IMREAD_UNCHANGED)
sw, sh, channels = src.shape
dw = int(sw/5)
dh = int(sh/5)
#inter = cv2.INTER_LINEAR
# 双线性插值。速度为最近邻和双三次的适中，效果也为二者适中。
inter = cv2.INTER_AREA
# 区域插值，共分三种情况。图像放大时类似于双线性插值，图像缩小(x轴、y轴同时缩小)又分两种情况，均可避免波纹出现。
#inter = cv2.INTER_LANCZOS4
# 兰索斯插值。8*8，公式类似于双线性，计算量更大，效果更好，速度较慢。
#iner = cv2.INTER_NEAREST
# 最近邻插值。因为没有插值，所以边缘会有严重的锯齿，放大后的图像有很严重的马赛克，缩小后的图像有很严重的失真。
#inter = cv2.INTER_CUBIC
# 双三次插值。有效地避免出现锯齿现象，但速度最慢。放大时效果最好，但速度很慢。实际测试中并不慢，可能是优化的原因。

#dst = cv2.resize(src, (d_h, d_w), interpolation=inter)
dst = cv2.resize(src, (0,0), fx=0.2, fy=0.2, interpolation=inter)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# area插值效果最好，5种算法中只有area可以看清车牌并且没有车牌边缘没有出现波纹，其他都多少出现扭曲。
# %%
