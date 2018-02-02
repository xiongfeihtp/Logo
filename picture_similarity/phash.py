import cv2
import numpy as np
import sys
import itertools
dim=32
step=4
def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    img=cv2.imread(imgfile, 0)
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)

        #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(dim,dim)
    img_list=list(itertools.chain.from_iterable(vis1.tolist()))
    #把二维list变成一维list
    # img_list=flatten(vis1.tolist())


    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+step]),2) for x in range(0,dim*dim,step)])

def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

#example for two picture
HASH1=pHash('./image_pinganlogo/an1.png')
HASH2=pHash('./image_pinganlogo/an2.png')
out_score1 = 1 - hammingDist(HASH1,HASH2)*1. / (dim*dim/step)


HASH1=pHash('./image_pinganlogo/an2.png')
HASH2=pHash('./image_pinganlogo/an3.png')

out_score2 = 1 - hammingDist(HASH1,HASH2)*1. / (dim*dim/step)

HASH1=pHash('./image_pinganlogo/an2.png')
HASH2=pHash('./image_pinganlogo/guo.jpeg')

out_score3 = 1 - hammingDist(HASH1,HASH2)*1. / (dim*dim/step)

print(out_score1)
print(out_score2)
print(out_score3)





