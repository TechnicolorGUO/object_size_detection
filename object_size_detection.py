import cv2
import numpy as np

def show(image):
    cv2.imshow("temp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def store(image,path):
    cv2.imwrite(path, image)

def obj_size_v1(path):
    '''
    Plain method for object size detection:
        1. Use red mask to localize the red points and determine the measurement.
        2. Localize the object with Canny edge detection and finding countours function.
        3. Calculate the size of the object with the measurement.  
    '''
    #Read the image
    img = cv2.imread(path)

    #Convert the image into HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Difine the range of the color RED
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # create the mask for the red dot
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Operate on the mask
    kernel_0 = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel_0, iterations=1)

    # Find the contours
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Draw the contours
    green_color = (0, 255, 0)
    distance = None

    point1 = None
    point2 = None
    maxY = -1

    for i, contour in enumerate(contours):
        # Draw the contours
        cv2.drawContours(img, [contour], 0, green_color, 2)
        
        # Find the kernal
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            cv2.circle(img, (cX, cY), 5, green_color, -1)
            
            if cY > maxY:
                maxY = cY
                point1 = (cX, cY)

            if i == 1:
                point2 = (cX, cY)

    cv2.line(img, point1, point2, green_color, 2)

    # Caculate the distance between 2 red dots
    distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    text = "{:.1f} cm".format(15)  
    cv2.putText(img, text, (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, green_color, 2)

    #-------------------------------------------------Object localization----------------------------------------------------

    #BGT 2 Gray
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Blur
    blur_image = cv2.bilateralFilter(gray_image,9,75,75)

    #Gray 2 canny with edge & Erase the area below higher red point
    canny_image = cv2.Canny(blur_image, 0, 200)
    canny_image = canny_image[0:maxY,:]

    #Dilate and erode
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(canny_image, kernel, iterations=1)
    erode_img = cv2.erode(dilated_img, kernel, iterations=1)

    #Fill the edge 
    filled_image = np.zeros_like(canny_image)
    contours, _ = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)

    #Find the contours of the filled-edge image
    original = img.copy()
    cnts = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #Find the box with the largest area
    max_area = 0
    max_contour = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_contour = c

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
                # Measure the dimensions of the bounding rectangle
        width = round((w/distance)*15, 2)
        height = round((h/distance)*15, 2)
        
        # Add the dimensions to the image
        cv2.putText(img, f'width: {width} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2)
        cv2.putText(img, f'height: {height} cm', (x + w, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2)
        ROI = original[y:y + h, x:x + w]

    show(img)
    store(img,"v1.jpg")




def obj_size_v2(path):

    '''
    Improved method for image with confusing background colors:
        1. Use red mask to localize the red points and determine the measurement.
        2. Gain the first binary image of contour with Canny algorithm
        3. Gain the second binary image of contour with adaptive threshold
        4. Overlay 2 binary images and then localize the object.
        5. Calculate the size of the object with the measurement.
    '''

    #Read the image
    img = cv2.imread(path)

    #Convert the image into HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Difine the range of the color RED
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # create the mask for the red dot
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Operate on the mask
    kernel_0 = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel_0, iterations=1)

    # Find the contours
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Draw the contours
    green_color = (0, 255, 0)
    distance = None

    point1 = None
    point2 = None
    maxY = -1

    for i, contour in enumerate(contours):
        # Draw the contours
        cv2.drawContours(img, [contour], 0, green_color, 2)
        
        # Find the kernal
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            cv2.circle(img, (cX, cY), 5, green_color, -1)
            
            if cY > maxY:
                maxY = cY
                point1 = (cX, cY)

            if i == 1:
                point2 = (cX, cY)

    cv2.line(img, point1, point2, green_color, 2)

    # Caculate the distance between 2 red dots
    distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    text = "{:.1f} cm".format(15)  
    cv2.putText(img, text, (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, green_color, 2)

    # -------------------------------------------------Object localization----------------------------------------------------



    '''Binary image 1: Use canny to detect the edge and then fill'''

    #BGT 2 Gray
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Blur
    blur_image = cv2.bilateralFilter(gray_image,9,75,75)

    #Gray 2 canny with edge
    canny_image = cv2.Canny(blur_image, 0, 200)
    canny_image = canny_image[0:maxY,:]

    #Dilate and erode edge image
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(canny_image, kernel, iterations=1)
    erode_img = cv2.erode(dilated_img, kernel, iterations=1)

    #Fill the edge 
    filled_image = np.zeros_like(canny_image)
    contours_canny, _ = cv2.findContours(erode_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled_image, contours_canny, -1, 255, thickness=cv2.FILLED)




    '''Binary image 2: Use threshold'''

    #Use adaptive threshold for gray 2 binary
    thresholded_image = cv2.adaptiveThreshold(gray_image[0:maxY,:], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    conv_image = cv2.erode(thresholded_image, (3,3), iterations=3)
    gau_image =cv2.bilateralFilter(conv_image,9,75,75)
    contours_thre, _ = cv2.findContours(gau_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    

    '''Add two binary image up'''
    re_image = np.logical_or(conv_image, filled_image)
    re_image = re_image.astype(np.uint8) * 255
    re_image = cv2.bilateralFilter(re_image,9,75,75)
    re_image = cv2.erode(re_image, (5,5), iterations=2)

    #Draw contours of both thresholded and canny binary images
    cv2.drawContours(re_image, contours_thre, -1, 255, thickness=1)
    cv2.drawContours(re_image, contours_canny, -1, 255, thickness=1)

    #Find the box with the largest area
    original = img.copy()
    cnts, _ = cv2.findContours(re_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_contour = c

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
        
        # Measure the dimensions of the bounding rectangle
        width = round((w/distance)*15, 2)
        height = round((h/distance)*15, 2)
        
        # Add the dimensions to the image
        cv2.putText(img, f'width: {width} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2)
        cv2.putText(img, f'height: {height} cm', (x + w, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2)
        ROI = original[y:y + h, x:x + w]

    show(img)
    store(img,"v2.jpg")
    

if __name__ == "__main__":
    # Path to your image
    image_path = "sample1.jpg"

    # Call the obj_size_v1 function
    obj_size_v1(image_path)
    obj_size_v2(image_path)