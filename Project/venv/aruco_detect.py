import cv2
import numpy as np
from utils import resize, Canny, Canny2, rgb2gray, canny, Canny_detector, visualize, harris, find_harris_corners, hough
from sift import harris_corner_detector
# from scipy.misc.pilutil import imread
# from imageio import imread
from cv2 import imread
# from matplotlib.pyplot import imread
# from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy import signal

def detect_aruco(img, dict_aruco):
    return 

def define_arucos(img):
    return 


def draw_lines(img, lines, color=(255, 0, 0)):
    img = img.copy()
    w, h = img.shape[:2]
    for line in lines:
        rho, theta, _ = line
        # if np.pi/2-0.5 >theta>np.pi/2+0.5 or 3*np.pi/2-0.5 >theta>3*np.pi/2+0.5:
        if -2 <theta<2 or 88<theta<92 or theta>178:
            a = np.cos(theta)
            b = np.sin(theta)
            m = b/a
            x0 = 0 
            y0 = b * rho
            print(theta,rho,a,b,x0,y0)
            x1 = w
            y1 = int(y0 + w * m)
            print(x1,y1)
            cv2.line(img, (x0, y0), (x1, y1), color, 2)
    return img


from scipy.ndimage import sobel
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import convolve2d
from scipy.optimize import curve_fit

# Define a function to compute the Hough Transform
def hough_transform(img):
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)) )
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * int(diag_len), num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

        # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho for each theta
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

# Define a function to detect lines from the Hough Transform
def detect_lines(img, threshold=100):
    edges = img.copy()
    # edges = sobel(img)
    # edges = binary_dilation(edges)
    accumulator, thetas, rhos = hough_transform(edges)
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    lines = []
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i,j]>threshold:
                line = [i-diag_len, j, thetas[j]]
                lines.append(line)
    return lines

# Use the detect_lines function to filter keypoints
# lines = detect_lines(img)
# keypoints_on_lines = []
# for keypoint in keypoints:
#     for line in lines:
#         rho, theta, _ = line
#         x,y = keypoint
#         if abs(x*cos(theta) + y*sin(theta) - rho) < threshold:
#             keypoints_on_lines.append(keypoint)
#             break


def main(): 
    tmp = cv2.imread(r"InitialDataset\InitialDataset\templates\template1_manyArucos.png")
    tmp_2 = tmp.copy()
    tmp = rgb2gray(tmp)
    print(tmp.size)
    tmp = resize(tmp,4)
    sigma = 0.5
    T = 0.3
    # canny = Canny(T, sigma)
    edg = Canny_detector(tmp)
    # edg = edg.astype(float)
    lines = detect_lines(edg)
    image_with_lines = draw_lines(tmp_2, lines)
    plt.imshow(image_with_lines)
    plt.show()
    
    # hough_img, t, r = hough(edg)
    # Example usage
    # img = cv2.imread(r"Project\venv\InitialDataset\InitialDataset\templates\template1_manyArucos.png")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    keypoints = harris_corner_detector(edg)
    print(edg.size)
    # canny = Canny2(sigma=0.5,kernel_size = 3)
    # edg = canny.edge_detect(tmp)
    
    
    # cv2.imshow("tmp",tmp)
    # cv2.imshow("tmp2",edg)
    # visualize(edg, 'gray')
    # im3,g_dx2,g_dy2,dx,dy,loc = harris(edg, im3= tmp_cv2)
    # corner_list, output_img = find_harris_corners(edg, tmp_cv2)
    # cv2.imshow("tmp",resize(tmp,4))
    # cv2.imshow("tmp2",resize(hough_img,4))
    cv2.waitKey(0)
    cv2.destroyAllWindows
    
if __name__ == "__main__":
    main()