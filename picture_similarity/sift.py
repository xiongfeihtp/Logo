# coding=utf-8
import cv2
import scipy as sp
#参数设置为0的话，默认读取的是灰度图
img1 = cv2.imread('./image_pinganlogo/an2.png', 0)  # queryImage
img2 = cv2.imread('./image_pinganlogo/an3.png', 0)  # trainImage

img2_reverse = img2.copy()
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        img2_reverse[i, j] = 255 - img2[i, j]

# Initiate SIFT detector
sift = cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img2_reverse, None)
# img1_ = cv2.drawKeypoints(img1, kp1, img1)
# cv2.imshow("img", img1_)
#
# cv2.waitKey()
# img2_ = cv2.drawKeypoints(img2, kp2, img2)
# cv2.imshow("img", img2_)
# cv2.waitKey()
# matching and give the line
# FLANN parameters

def give_point(one,two):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(one, two, k=2)
    return matches

def plot(img1, img2, good):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    for m in good1:
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple([sp.random.randint(0, 255) for _ in range(3)])
        # print 'kp1,kp2',kp1,kp2
        cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
                 (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color)
    cv2.imshow("view", view)
    cv2.waitKey()

matches1=give_point(des1,des2)
matches2=give_point(des1,des3)
print('matches...raw', len(matches1))
# Apply ratio test
good1 = []
for m, n in matches1:
    if m.distance < 0.65 * n.distance:
        good1.append(m)
print('good', len(good1))


print('matches...reverse', len(matches1))
# Apply ratio test
good2 = []
for m, n in matches1:
    if m.distance < 0.5 * n.distance:
        good2.append(m)
print('good', len(good2))

plot(img1,img2,good1)
plot(img1,img2_reverse,good2)



