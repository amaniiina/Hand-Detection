from os import listdir
from os.path import isfile, join
import cv2
from PIL import Image


# if file is an image return true else return false
def isImage(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True


# return list of all images in path
def getImgsFromPath(folderPath, gray=0):
    images = []
    for f in listdir(folderPath):
        imgPath = join(folderPath, f)
        if isfile(imgPath) and isImage(imgPath):
            images.append(cv2.imread(imgPath, gray))
    return images


def gaussianBlur(img, kernelSize=3, sigma=0, iterations=1):
    for i in range(iterations):
        img = cv2.GaussianBlur(img.copy(), (kernelSize, kernelSize), sigma)
    return img


def medianBlur(img, kernelSize=3, iterations=1):
    for i in range(iterations):
        img = cv2.medianBlur(img, kernelSize)
    return img


def getPointNum(boxNum):
    if boxNum == 6:
        return 7
    elif boxNum == 8:
        return 6
    elif boxNum == 10 or boxNum == 11:
        return 9
    elif boxNum == 13:
        return 4
    elif boxNum == 14:
        return 3
    elif boxNum == 16 or boxNum == 17:
        return 1
    elif boxNum == 18:
        return 2
    else:
        return -1
