# ORB
import cv2
import numpy as np
from matplotlib import pyplot as plt
from colordescriptor import ColorDescriptor


def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])
    # return the chi-squared distance
    return d
# readImage
img1 = cv2.imread("./image_pinganlogo/pingan1.jpg")  # queryImage
img2 = cv2.imread("./image_pinganlogo/pingan3.jpeg")  # trainImage

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))
hoc1 =cd.describe(img1)
hoc2 =cd.describe(img2)
hoc_distance=chi2_distance(hoc1,hoc2)


# img2_reverse = img2.copy()
# for i in range(img2.shape[0]):
#     for j in range(img2.shape[1]):
#         img2_reverse[i, j] = 255 - img2[i, j]
# img2=img2_reverse
def Binarization(img):
    GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 中值滤波
    GrayImage= cv2.medianBlur(GrayImage,5)
    ret,th1 = cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
    #3 为Block size, 5为param1值
    th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                        cv2.THRESH_BINARY,3,5)
    th3 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,3,5)
    images = [GrayImage, th1, th2, th3]
    # titles = ['Gray Image', 'Global Thresholding (v = 127)',
    #           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    return images

def extractBestMatches(matches):
    minDist = 500
    maxDist = 0
    for item in matches:
        distance = item.distance
        if distance > 0:
            if distance < minDist:
                minDist = distance
            if distance > maxDist:
                maxDist = distance
    threshold = 3 * minDist
    goodmatches = []
    for item in matches:
        distance = item.distance
        if distance < threshold:
            goodmatches.append(item)
    return goodmatches


def calculateScore(goodmatches, kp1, kp2):
    row_kp1=len(kp1)
    row_kp2=len(kp2)
    row_matches=len(goodmatches)
    denom=row_kp1*row_kp2
    numer=(row_kp1-row_matches)*(row_kp2-row_matches)
    if denom!=0:
        return 1-(numer/denom)
    else:
        return -1

images1=Binarization(img1)
images2=Binarization(img2)
for img1,img2 in zip(images1,images2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # kp3, des3 = orb.detectAndCompute(img2_reverse,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    try:
        matches = bf.match(des1, des2)
    except Exception as e:
        continue
    bestMatches=extractBestMatches(matches)
    score=calculateScore(bestMatches,kp1,kp2)
    print("match score:{}".format(score))
    #Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    plt.imshow(img3), plt.show()

