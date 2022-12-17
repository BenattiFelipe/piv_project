import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import draw_corners, resize, corresp_points, know_template
from algebra import find_homography, ransac

def main():
    # vid_path = "InitialDataset\InitialDataset\ManyArucos.mp4"
    vid_path = "InitialDataset\InitialDataset\FewArucos-Viewpoint2.mp4"
    vid = cv2.VideoCapture(vid_path)
    ret, image = vid.read()
    template = know_template(image)
    wt, ht = template.shape[:2]
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(template, arucoDict,
        parameters=arucoParams)
    tmp_prt = draw_corners(corners, ids, template)
    tmp_prt = resize(tmp_prt, 4)
    dic_temp = dict(zip(ids.flatten(), corners))
    # cv2.imshow('template', tmp_prt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    while True:
        # Capture frame-by-frame
        ret, image = vid.read()
        height, width = image.shape[:2]
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
            parameters=arucoParams)
        dic_img = dict(zip(ids.flatten(), corners))
        pi, pt = corresp_points(dic_img, dic_temp)
        pi = np.array(pi)
        pt = np.array(pt)
        print("pi",pi,"pt",pt)
        h, status = cv2.findHomography(pi, pt, cv2.RANSAC)
        # H = find_homography(pi[:4], pt[:4])
        H, _ = ransac(pi,pt)
        # print(H,h)
        im_dst = cv2.warpPerspective(image, H, (ht, wt))
        im_dst = resize(im_dst, 4)
        cv2.imshow("dst", im_dst)
        
        image = draw_corners(corners, ids, image)
        image = cv2.resize(image, (width//2, height//2), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('vid', image)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    

# '''
# pts_src and pts_dst are numpy arrays of points
# in source and destination images. We need at least
# 4 corresponding points.
# '''
# h, status = cv2.findHomography(pts_src, pts_dst)
 
# '''
# The calculated homography can be used to warp
# the source image to destination. Size is the
# size (width,height) of im_dst
# '''
 
# im_dst = cv2.warpPerspective(im_src, h, size)