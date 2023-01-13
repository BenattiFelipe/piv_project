import cv2
import numpy as np
import math
import os

def know_template(image):
    template_many = cv2.imread(r"InitialDataset\InitialDataset\templates\template1_manyArucos.png")
    template_few = cv2.imread(r"InitialDataset\InitialDataset\templates\template2_fewArucos.png")
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
        parameters=arucoParams)
    if len(ids.flatten())>4:
        return template_many
    else:
        return template_few

# TODO : func check template
def resize(img, r):
    height, width = img.shape[:2]
    img = cv2.resize(img, (width//r, height//r), interpolation=cv2.INTER_CUBIC)
    return img

def draw_corners(corners, ids, frame, name_id=True):
	# verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the frame
            if name_id:
                cv2.putText(frame, str(markerID),
                    (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return(frame)

def corresp_points(dic_img, dic_temp):
    corri = []
    corrt = []
    for k in dic_img.keys():
        ci = dic_img[k]
        ct = dic_temp[k]
        for i in range(len(ci)):
            print("ci", ci[0][i])
            corri.append(ci[0][i])
            corrt.append(ct[0][i])
    return([corri,corrt])

def find_corr(pt1, pt2):
    corr = np.zeros((len(pt2),4))
    for i in range(len(corr)):
        corr[i] = [pt1[i,0], pt1[i,1],
                   pt2[i,0], pt2[i,1]]
    return corr
        
def detect_points(img):
    return 0
    
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


from scipy import ndimage
from scipy.ndimage.filters import convolve, gaussian_filter

def Canny_detector(img,highThreshold=91,lowThreshold=31):
  img=np.array(img,dtype=float) #convert to float to prevent clipping values

  #step1: Noise reduction
  img=gaussian_filter(img,sigma=0.5,truncate=0.3)

  #step2: Calculating gradients
  Kx=[[-1,0,1],[-2,0,2],[-1,0,1]]  #Sobel filters
  Ky=[[1,2,1],[0,0,0],[-1,-2,-1]]
  Ix=convolve(img,Kx)
  Iy=convolve(img,Ky)
  grad=np.hypot(Ix,Iy) #magnitude of gradient
  theta=np.arctan2(Iy,Ix) #slope theta of gradient

  thetaQ=(np.round(theta*(5.0/np.pi))+5)%5 #Quantize direction

  #step3: Non-maximum suppression
  gradS=grad.copy()
  for r in range(img.shape[0]):
    for c in range(img.shape[1]):
      #suppress pixels at the image edge
      if r==0 or r==img.shape[0]-1 or c==0 or c==img.shape[1]-1:
        gradS[r,c]=0
        continue
      tq=thetaQ[r,c] % 4

      if tq==0: #0 is E-W (horizontal)
        if grad[r,c] <= grad[r,c-1] or grad[r,c]<=grad[r,c+1]:
          gradS[r,c]=0
      if tq==1: #1 is NE-SW
        if grad[r,c] <= grad[r-1,c+1] or grad[r,c]<=grad[r+1,c-1]:
          gradS[r,c]=0
      if tq==2: #2 is N-S (vertical)
        if grad[r,c] <= grad[r-1,c] or grad[r,c]<=grad[r+1,c]:
          gradS[r,c]=0
      if tq==3: #3 is NW-SE
        if grad[r,c] <= grad[r-1,c-1] or grad[r,c]<=grad[r+1,c+1]:
          gradS[r,c]=0

    #step4: Double threshold
    strongEdges=(gradS>highThreshold)
    #strong has value 2, weak has value 1
    thresholdEdges=np.array(strongEdges,dtype=np.uint8)+ (gradS>lowThreshold)

    #step5: Edge tracking by hysterisis
    #Find weak edge pixels near strong edge pixels
    finalEdges=strongEdges.copy()
    currentPixels=[]
    for r in range(1,img.shape[0]-1):
      for c in range(1,img.shape[1]-1):
        if thresholdEdges[r,c]!=1:
          continue #Not a weak pixel
          
        #get a 3X3 patch
        localPatch=thresholdEdges[r-1:r+2,c-1:c+2]
        patchMax=localPatch.max()
        if patchMax==2:
          currentPixels.append((r,c))
          finalEdges[r,c]=1
        
    #Extend strong edges based on current pixels
    while len(currentPixels) > 0:
      newPixels=[]
      for r,c in currentPixels:
        for dr in range(-1,2):
            for dc in range(-1,2):
              if dr==0 and dc==0:
                continue
              r2=r+dr
              c2=c+dc
              if thresholdEdges[r2,c2]==1 and finalEdges[r2,c2]==0:
                #copy this weak pixel to final result
                newPixels.append((r2,c2))
                finalEdges[r2,c2]=1
      currentPixels= newPixels
        
    return finalEdges

import numpy as np
from scipy.spatial.distance import cdist


def match_descriptors(descriptors1, descriptors2, metric=None, p=2,
                      max_distance=np.inf, cross_check=True, max_ratio=1.0):
    """Brute-force matching of descriptors.
    For each descriptor in the first set this matcher finds the closest
    descriptor in the second set (and vice-versa in the case of enabled
    cross-checking).
    Parameters
    ----------
    descriptors1 : (M, P) array
        Descriptors of size P about M keypoints in the first image.
    descriptors2 : (N, P) array
        Descriptors of size P about N keypoints in the second image.
    metric : {'euclidean', 'cityblock', 'minkowski', 'hamming', ...} , optional
        The metric to compute the distance between two descriptors. See
        `scipy.spatial.distance.cdist` for all possible types. The hamming
        distance should be used for binary descriptors. By default the L2-norm
        is used for all descriptors of dtype float or double and the Hamming
        distance is used for binary descriptors automatically.
    p : int, optional
        The p-norm to apply for ``metric='minkowski'``.
    max_distance : float, optional
        Maximum allowed distance between descriptors of two keypoints
        in separate images to be regarded as a match.
    cross_check : bool, optional
        If True, the matched keypoints are returned after cross checking i.e. a
        matched pair (keypoint1, keypoint2) is returned if keypoint2 is the
        best match for keypoint1 in second image and keypoint1 is the best
        match for keypoint2 in first image.
    max_ratio : float, optional
        Maximum ratio of distances between first and second closest descriptor
        in the second set of descriptors. This threshold is useful to filter
        ambiguous matches between the two descriptor sets. The choice of this
        value depends on the statistics of the chosen descriptor, e.g.,
        for SIFT descriptors a value of 0.8 is usually chosen, see
        D.G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
        International Journal of Computer Vision, 2004.
    Returns
    -------
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    """

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor length must equal.")

    if metric is None:
        if np.issubdtype(descriptors1.dtype, bool):
            metric = 'hamming'
        else:
            metric = 'euclidean'

    kwargs = {}
    # Scipy raises an error if p is passed as an extra argument when it isn't
    # necessary for the chosen metric.
    if metric == 'minkowski':
        kwargs['p'] = p
    distances = cdist(descriptors1, descriptors2, metric=metric, **kwargs)

    indices1 = np.arange(descriptors1.shape[0])
    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] \
            = np.finfo(np.float64).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]
            
    matches = np.column_stack((indices1, indices2))
    return matches



