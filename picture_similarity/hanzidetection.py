import cv2
def intersected(rc1, rc2):
    if rc1[0] > rc2[2]: return False
    if rc1[1] > rc2[3]: return False
    if rc2[0] > rc1[2]: return False
    if rc2[1] > rc1[3]: return False
    return True
grey_lim=200
def segment(grey):
    _, thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    #白色背景需要注释，镂空深色背景需要取反
    # for i in range(thresh.shape[0]):
    #     for j in range(thresh.shape[1]):
    #         thresh[i, j] = 255 - thresh[i, j]
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    _, countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rcs = map(cv2.boundingRect, countours)
    rcs = [(rc[0], rc[1], rc[0] + rc[2], rc[1] + rc[3]) for rc in rcs]
    # change the countours
    # filter in rcs
    rcs_med = rcs.copy()
    max_lim=1024 * 0.8
    min_lim=1024 * 0.001
    for i, rc in enumerate(rcs_med):
        print(i)
        xlim = abs(rc[0] - rc[2])
        ylim = abs(rc[1] - rc[3])
        print(xlim,ylim)
        if xlim <max_lim and ylim <max_lim and xlim>min_lim and ylim>min_lim:
            pass
            #del 和 remove 的区别，这里del会有bug
        else:
            print('del')
            rcs.remove(rcs_med[i])
    # clustering
    clusters = list(range(len(rcs)))
    # 边框两两做比较,对不具有包含关系的边框进行类别标注
    for i, rc in enumerate(rcs):
        for j, irc in enumerate(rcs[i + 1:]):
            idx = i + j + 1
            if clusters[idx] != clusters[i] and intersected(rc, irc):
                if clusters[idx] > clusters[i]:
                    clusters[idx] = clusters[i]
                else:
                    clusters[i] = clusters[idx]
    # 定义了几类，但是都被最大的框吃掉了
    def cluster(v):
        indices = [i for i, x in enumerate(clusters) if x == v]
        xmin = min(rcs[idx][0] for idx in indices)
        ymin = min(rcs[idx][1] for idx in indices)
        xmax = max(rcs[idx][2] for idx in indices)
        ymax = max(rcs[idx][3] for idx in indices)
        return xmin, ymin, xmax, ymax

    rcs = list(map(cluster, set(clusters)))
    h = int(sum((rc[3] - rc[1]) for rc in rcs)) / len(list(rcs))
    w = thresh.shape[1]
    #按面积进行排序
    return sorted(rcs, key=lambda rc: int(rc[1]) / h * w + rc[0]), thresh
def main(path):
    im = cv2.imread(path)
    # segmentation
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY,0)
    cv2.imshow('img_reverse', grey)
    cv2.waitKey(0)
    rcs, thresh = segment(grey)
    # drawing
    draw = im.copy()
    for i, rc in enumerate(rcs):
        cv2.rectangle(draw, rc[0:2], rc[2:4], (0, 0, 0),3)
        cv2.putText(draw, str(i), (rc[0], rc[3]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        # cv2.drawContours(draw, countours, i, (255, 0, 0))
    cv2.imshow(path, draw)
    cv2.imwrite(path + '_draw.jpeg', draw)
    return thresh, rcs

if __name__ == '__main__':
    main('./image_pinganlogo/pinganjinzhou.jpeg')
    # main('./image_pinganlogo/pingan9.jpeg')
    cv2.waitKey(0)
