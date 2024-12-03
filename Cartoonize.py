import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(IMG_FILE):
    ori_img = cv2.imread(IMG_FILE) # BGR IMAGE
    rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    return rgb_img

def edge_mask(img, line_size, blur_size):
    gray_img =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # To GrayScale img
    gray_blur = cv2.medianBlur(gray_img, blur_size)  # median
    # gray_blur = cv2.blur(gray_img, (blur_size, blur_size)) # mean
    # gray_blur = cv2.GaussianBlur(gray_img, (blur_size, blur_size), 0) # guassian

    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_size)

    return edges

def color_quantization(img, k):
    
    data = np.float32(img).reshape((-1, 3))

    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001 # condition

    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result

def cartoon(img, k):
    img = read_file(IMG_FILE) # Change BGR to RGB (plt lib)
    edges = edge_mask(img, 9, 7) # Make edges mask
    img_quantized = color_quantization(img, k) # Quantized color img

    c = cv2.bitwise_and(img_quantized, img_quantized , mask =edges) # merged img and mask
    
    plt.figure()
    plt.imshow(c)
    plt.title("Cartoon Image")
    plt.show()

IMG_FILE = "PLACE IMAGE HERE.jpg"
cartoon(IMG_FILE, "add the Quantized number of image") 

