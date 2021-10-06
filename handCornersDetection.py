
import numpy as np
import cv2
import os
import xlwt
from extraFiles import functions


def main():
    path = './images'
    images_orig = functions.getImgsFromPath(path, 1)
    images = functions.getImgsFromPath(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if os.path.exists("./palmPoints.xls"):
        os.remove("./palmPoints.xls")
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Palm Points")
    sheet.write(0, 0, 'Point')
    sheet.write(0, 1, 'X')
    sheet.write(0, 2, 'Y')
    excel_row = 1

    for idx in range(len(images)):
        img = images[idx]
        orig = images_orig[idx]
        img = functions.medianBlur(img, 7, 20)
        img = cv2.Canny(img, 60, 130) #130
        img = cv2.dilate(img, kernel, iterations=1)
        # entropy = skimage.measure.shannon_entropy(orig)
        # print(entropy, img.shape)

        horizontalStep = img.shape[1] // 4
        verticalStep = img.shape[0] // 5
        boxNum = 0
        for horiz in range(0, img.shape[1]-1, horizontalStep):
            # skip extra pixels at end of image width
            if boxNum > 19:
                break
            # skip first column (fingertips)
            if 0 < boxNum < 5:
                boxNum = 5
                continue
            verticalSliceNum = 0
            for verti in range(0, img.shape[0]-1, verticalStep):
                # skip extra pixels at end of image length
                if verticalSliceNum > 4:
                    continue
                sliced = img[verti:verti+verticalStep, horiz:horiz+horizontalStep]
                # print(boxNum, cv2.countNonZero(sliced))

                # skip boxes (0,1),(3,1),(2,2),(0,3),(3,3) (don't include any needed points)
                if boxNum == 5 or boxNum == 7 or boxNum == 9 or boxNum == 10 or boxNum == 12 or boxNum == 15 or \
                        boxNum == 19:
                    boxNum += 1
                    verticalSliceNum += 1
                    continue
                else:
                    # use good features to track to find strongest corner in each part of image
                    corners = cv2.goodFeaturesToTrack(sliced[:, :sliced.shape[1]-38], 1, 0.01, 5, useHarrisDetector=True)
                    if corners is not None:
                        corners = np.int0(corners)
                        for i in corners:
                            x, y = i.ravel()
                            if 4 < x < horiz-4 and 4 < y < verti-4:
                                x = x+horiz
                                y = y+verti
                                cv2.circle(orig, (x, y), 8, (0, 0, 255), -1)
                                sheet.write(excel_row, 0, str(functions.getPointNum(boxNum)))
                                sheet.write(excel_row, 1, str(x))
                                sheet.write(excel_row, 2, str(y))
                                excel_row += 1
                verticalSliceNum += 1
                boxNum += 1
        excel_row += 1
        cv2.imshow('result', orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    workbook.save("palmPoints.xls")


if __name__ == "__main__":
    main()
