import cv2
import numpy as np
from utils import resize, rgb2gray, Canny_detector, match_descriptors
from sift import match_features, orb
# from scipy.misc.pilutil import imread
# from imageio import imread
from cv2 import imread
# from matplotlib.pyplot import imread
# from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy import signal
from algebra import ransac
import pysift

def detect_aruco(img, dict_aruco):
    return 

def define_arucos(img):
    return 

def draw_keypoints(img, keypoints):
    # Create a copy of the image
    img_keypoints = img.copy()
    for x, y in keypoints:
        img_keypoints[x-3:x+3, y-3:y+3] = [255, 0, 0]
    plt.imshow(img_keypoints, cmap='gray')
    plt.show()


import numpy as np
from scipy.ndimage.filters import convolve
from scipy.spatial import cKDTree

def bf(kp1, des1, kp2, des2):
    from scipy.spatial.distance import cdist
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    # compute pairwise distances between the descriptors
    distances = cdist(des1, des2)

    # get indices of the nearest neighbors for each descriptor
    indices = np.argmin(distances, axis=1)
    # indices = indices.astype(np.int32)
    print(indices)
    # indices = np.random.choice(indices, 100)
    # extract the corresponding keypoints
    matched_kp1 = kp1[indices]
    matched_kp2 = kp2[indices]
    
    return matched_kp1, matched_kp2, indices

def Kd(kp1, des1, kp2, des2):
    # Create a KD-Tree from the second set of keypoints
    kdt = cKDTree(des2)
    # Find the nearest neighbor for each keypoint in the first image
    print(des1.shape)
    distances, indices = kdt.query(des1, distance_upper_bound=1e-5)
    # Filter out any invalid matches (indices of -1)
    print(indices)
    valid_indices = indices[indices != -1]
    # Extract the corresponding keypoints
    # print(valid_indices.shape, valid_indices.max(),kp1.shape)
    print(valid_indices)
    valid_indices.astype(np.int)
    matched_kp1 = kp1[valid_indices]
    matched_kp2 = kp2[valid_indices]
    return matched_kp1, matched_kp2


def match_features_2(kp1, kp2, des1, des2):
    # # # Compute the Canny edge maps
    # edges1 = Canny_detector(img1)
    # edges2 = Canny_detector(img2)
    # print(edges1.shape)

        # Extract keypoints from the edge maps using the Harris corner detector
    # draw_keypoints(i1,kp1)
    # draw_keypoints(i2,kp2)
    # des1 = des1.reshape((-1,1))
    # des2 = des2.qreshape((-1,1))
    if des1.shape[1] > des2.shape[1]:
        des1 = des1[:, :des2.shape[1]]
    else:
        des2 = des2[:, :des1.shape[1]]
    matched_kp1, matched_kp2, indices = bf(kp1, des1, kp2, des2)
    return matched_kp1, matched_kp2, indices, kp1, kp2

import matplotlib.pyplot as plt

def draw_matches(img1, kp1, img2, kp2, indices):
    # Create a new image to show the matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    img_matches[:img1.shape[0], :img1.shape[1]] = img1
    img_matches[:img2.shape[0], img1.shape[1]:] = img2
    m1 = m2 = []
    for i in range(len(indices)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        p1 = tuple(map(int, kp1[i][::-1]))
        p2 = tuple(map(int, kp2[indices[i]][::-1]))
        m1.append(list(p1))
        m2.append(list(p2))
        x_off = img1.shape[1]
        #draw a line between the points
        cv2.line(img_matches, p1, (p2[0]+x_off, p2[1]), color)
    # plt.imshow(img_matches)
    # plt.show()
    return m1,m2, img_matches


def harris(img):
    # Define the Harris corner detector kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    # Compute the Harris corner response
    Ix = convolve(img.astype(np.float32), np.array([[1, 0, -1]], dtype=np.float32))
    Iy = convolve(img.astype(np.float32), np.array([[1], [0], [-1]], dtype=np.float32))
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    R = convolve(Ixx, kernel) * convolve(Iyy, kernel) - (convolve(Ixy, kernel) ** 2)
    # Extract the keypoints
    keypoints = np.column_stack(np.where(R > R.max() * 0.1))
    descriptor = np.array(R)
    return keypoints, descriptor

from scipy.interpolate import griddata
import numpy as np

def warpPerspective(img, M, dsize):
    # create the grid of points to warp
    h, w = img.shape[:2]
    x, y = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
    x_map, y_map = np.meshgrid(x, y)
    ones = np.ones((h, w))
    src_points = np.dstack((x_map, y_map, ones))

    # apply the perspective transformation
    dst_points = src_points @ np.transpose(M)
    dst_points = dst_points/dst_points[:,:,-1:]

    # get the target image size
    h_dst, w_dst = dsize
    x_dst, y_dst = np.linspace(0, w_dst-1, w_dst), np.linspace(0, h_dst-1, h_dst)
    x_map_dst, y_map_dst = np.meshgrid(x_dst, y_dst)

    # Interpolate the image
    img_warped = griddata((dst_points[:,:,0].ravel(), dst_points[:,:,1].ravel()), img.ravel(), (x_map_dst, y_map_dst), method='linear')

    return img_warped


def less_points(kp):
    dist = lambda x1,y1,x2,y2 : np.sqrt((x1-x2)**2+(y1-y2)**2)
    return

def clean_prox(x,ind):
    new = []
    for i in range(len(x)-1):
        if np.abs(x[i]-x[i+1])<2:
            new.append(ind[i])
    return new

def find_points(kp):
    X = kp[:,0]
    ind_x = sorted(range(len(X)), key=lambda k: X[k]) 
    x_sort = sorted(X)
    Y = kp[:,1]
    ind_y = sorted(range(len(Y)), key=lambda k: Y[k]) 
    y_sort = sorted(Y)
    dist = lambda x1,y1,x2,y2 : np.sqrt((x1-x2)**2+(y1-y2)**2)
    xmin = x_sort[0]
    xmax = x_sort[-1]
    ymin = y_sort[0]
    ymax = y_sort[-1]
    pxmin = kp[ind_x[0]]
    pxmax = kp[ind_x[-1]]
    p = kp[[ind_x[-8:]]]
    p2 = kp[[ind_x[0:8]]] 
    # pymin = kp[ind_y[0]]
    # pymax = kp[ind_y[-1]]
    print(pxmin, pxmax)#, pymin, pymax)
    return p#, pymin, pymax]
    
    
    
    

def main(): 
    tmp = cv2.imread(r"InitialDataset\InitialDataset\templates\template1_manyArucos.png")
    vid = cv2.VideoCapture(r"InitialDataset\InitialDataset\ManyArucos.mp4")
    # cv2.imshow("temp", tmp)
    # cv2.waitKey(0)
    tmp = resize(tmp,4)
    tmp = tmp.astype(np.float32)
    tmpg = rgb2gray(tmp)
    tmpg = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    # print(tmpg.shape)
    kp1, des1 = harris(tmpg)
    ps = find_points(kp1)
    draw_keypoints(tmp, ps)
    # kp1, des1 = pysift.computeKeypointsAndDescriptors(tmpg)
    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(tmpg,None)
    cont = 0
    # while True:
    #     ret, img = vid.read()
    #     # img = resize(img,4)
    #     # img = img.astype(np.float32)
    #     # imgg = rgb2gray(img)
    #     imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # sigma = 0.5
    #     # T = 0.3
    #     # canny = Canny(T, sigma)
    #     # d1 = orb(tmp)
    #     # d2 = orb(img)
    #     # ind = match_features(d1,d2)
    #     # kp2, des2 = harris(imgg)
    #     # kp2, des2 = pysift.computeKeypointsAndDescriptors(imgg)
    #     kp2, des2 = sift.detectAndCompute(imgg,None)
    #     # matches = match_descriptors(des1, des2)
    #     # m1, m2 = Kd(kp1, des1, kp2, des2)
    #     # print(matches[:, 0], matches[:,1])
    #     # matched_kp1, matched_kp2, indices, kp1, kp2 = match_features_2(kp1, kp2, des1, des2)
    #     # m1,m2, img_m = draw_matches(tmp, kp1, img, kp2, indices)
    #     # Initialize and use FLANN
    #     FLANN_INDEX_KDTREE = 0
    #     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #     search_params = dict(checks = 50)
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     # matches = flann.knnMatch(des1, des2, k=2)
    #     matches = flann.knnMatch(des1, des2, k=2)
    #     # print(m1[0],m2[0])
    #     # print(matched_kp1[:2],matched_kp2[:2])
    #     # print(matches)
    #     good = []
    #     for m, n in matches:
    #         good.append(m)
    #         # Estimate homography between template and scene
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good])#.reshape(-1, 1, 2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good])#.reshape(-1, 1, 2)
            
    #     # print(src_pts)
    #     # print(dst_pts)
    #     # H, m = ransac(matches, [], thresh = 5.,max=500 , bool=0)
    #     H, m = ransac(dst_pts,src_pts, thresh = 5 ,max=1000)
    #     # H = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)[0]
    #     # w_img = warpPerspective(img,H,tmp.shape[:2])
    #     w, h = tmp.shape[:2]
    #     w_img = cv2.warpPerspective(img, H, (h,w))
    #     # plt.imshow(w_img)
    #     # plt.show()
    #     w_img = resize(w_img,4)
    #     cv2.imshow('vid', w_img)
    #     # cv2.imshow('vid', img_m)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #     # edg = Canny_detector(tmp)
    #     # print(edg.size)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows
        
if __name__ == "__main__":
    main()