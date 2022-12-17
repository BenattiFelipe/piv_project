import cv2
import numpy as np

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
        