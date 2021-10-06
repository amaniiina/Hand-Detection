
import cv2
import skimage.measure
from extraFiles import functions


def main():
    path = './images'
    images_orig = functions.getImgsFromPath(path, 1)
    images = functions.getImgsFromPath(path)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for idx in range(len(images)):
        img = images[idx]
        entropy = skimage.measure.shannon_entropy(img)
        print(idx, entropy)
        if entropy < 6.5:
            img = cv2.equalizeHist(img)
            img = functions.gaussianBlur(img, 21, 1, 3)
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=1)
            thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1] #210,255
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        else:
            img = functions.gaussianBlur(img, 21, 0, 2)
            thresh = cv2.threshold(img, 183, 255, cv2.THRESH_BINARY)[1] #190, 255
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
            thresh = cv2.dilate(thresh, kernel, iterations=4)
        cnts, hierarchy = cv2.findContours(thresh[:, 5:int(thresh.shape[1] / 2.4)],
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) != 0:
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                key=lambda b: b[1][2]*b[1][3], reverse=True))
            numContours = 5 if len(cnts) >= 5 else len(cnts)
            for j in range(numContours):
                # compute the center of the contour area and draw a circle representing the center
                x, y, w, h =boundingBoxes[j]
                M = cv2.moments(cnts[j])
                if M["m00"] != 0:
                    centerX = int(M["m10"] / M["m00"])
                    centerY = int(M["m01"] / M["m00"])
                    # subtract from center to make circle closer to fingertips (move along diagonal)
                    cv2.circle(images_orig[idx], (centerX-(w//5), centerY-(h//5)), 6, (0,0,255), -1)

        cv2.imshow('result', images_orig[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
