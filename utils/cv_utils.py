import cv2
from matplotlib import pyplot as plt
import numpy as np
def align_face(image):
    eye_detector = cv2.CascadeClassifier("./haarcascade_eye.xml")
    aligned_face=[]
    if image.shape[0]<100:
        for i in range(image.shape[0]):
            gray_face = cv2.cvtColor(image[i], cv2.COLOR_BGR2GRAY)
            eyes = eye_detector.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            right_eye_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            left_eye_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]) * 180 / np.pi
            M = cv2.getRotationMatrix2D((112 / 2, 112 / 2), angle, 1)
            aligned_face = cv2.warpAffine(image[i], M, (112, 112))
        aligned_face=torch.stack(aligned_face)
        aligned_face=aligned_face.cuda()
    else:
        gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if type(eyes) == tuple or eyes.shape[0]!=2:
            aligned_face = image
            #imageio.imwrite('/fs1/home/tjuvis_2022/lxx/DFRF-main/NeRF-pre/test/'+str(eyes.shape[0])+'.jpg', image)
        else:
            eyes = sorted(eyes, key=lambda x: x[0])
            #print(eyes)
            right_eye_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            left_eye_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]) * 180 / np.pi
            M = cv2.getRotationMatrix2D((112 / 2, 112 / 2), angle, 1)
            aligned_face = cv2.warpAffine(image, M, (112, 112))        
    return aligned_face
def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)
    #if img.shape[0] <450:
    #    img=align_face(img)
    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_row(imgs, titles, rows=1):
    '''
       Display grid of cv2 images image
       :param img: list [cv::mat]
       :param title: titles
       :return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        plt.axis('off')
    plt.show()