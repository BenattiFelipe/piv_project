import cv2
import numpy as np
from utils import find_corr

def find_error(c, h):
    p1 = np.array([c[0], c[1], 1]).T
    p2 = np.array([c[2], c[3], 1]).T
    p2_1 = np.dot(h, p1)
    p2_1 = (1/p2_1[0,2])*p2_1
    error = p2 - p2_1
    return np.linalg.norm(error)

def ransac(pt1,pt2, thresh = 5.):
    corr = find_corr(pt1, pt2)
    maxInliers = []
    finalH = None
    for i in range(len(pt1)**2):
        #find 4 random points to calculate a homography
        corr1 = corr[np.random.randint(0, len(corr))]
        corr2 = corr[np.random.randint(0, len(corr))]
        corr3 = corr[np.random.randint(0, len(corr))]
        corr4 = corr[np.random.randint(0, len(corr))]
        randomFour = np.vstack((corr1, corr2, corr3, corr4))
        pt1 = randomFour[:,:2]
        pt2 = randomFour[:,2:]
        h = find_homography(pt1,pt2)
        inliers = []

        for i in range(len(corr)):
            d = find_error(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        # print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers

def find_homography(pt1, pt2, method=0):
    A = []
    # print("PI",pt1[0,1], pt2[0])
    for i in range(len(pt1)):
        p1 = np.array([pt1[i,0], pt1[i,1], 1])
        p2 = np.array([pt2[i,0], pt2[i,1], 1])
        # print("P",p1[0], p2[0])
        a2 = [0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
              p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]]
        a1 = [-p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
              p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]]
        A.append(a1)
        A.append(a2)
    matrixA = np.matrix(A)
    u, s, v = np.linalg.svd(matrixA)
    h = np.reshape(v[8], (3, 3))
    h = (1/h[2,2]) * h
    return h
