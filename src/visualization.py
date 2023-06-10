import cv2

def visualize_pts(imgpoints, optpoints, image, number):
    """
    a function to visualize the reporjected points
    :param imgpoints: image points
    :param optpoints: output points
    :param image: image
    :param number: image number
    """
    img = cv2.imread(image)
    for i in imgpoints:
        x = i[0]
        y = i[1]
        # x_correct = optpoints[i][0]
        # y_correct = optpoints[i][1]
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        # cv2.rectangle(img, (x_correct - 5, y_correct - 5), (x_correct + 5, y_correct + 5), (0, 255, 0), -1)
        cv2.imwrite("/Users/ppangppang/Documents/dev/cv/results/reproj_{}.jpg".format(number), img)
        
