import numpy as np
import glob
import cv2
import Utils

imagesPath = "Images\\"
resultImagesPath = "ResultImages\\"
scale = 0.5
color = (0,0,255)
widthKTP = 850
heightKTP = 540

def nothing(x):
    pass

def main():
    for filename in glob.glob(imagesPath+"*.jpg"):
        
        img = cv2.imread(filename)
        print(filename, img.shape)

        #############################
        img = Utils.resize(img, scale)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (15,15), 3)
        imgCanny = cv2.Canny(imgBlur, 22, 50)
        imgThresh = cv2.adaptiveThreshold(imgCanny, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,2)
        
        ##########################
        imgContours = img.copy()
        contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, color, 2)

        #########################
        rectanglePoints = Utils.findRectangle(contours)

        #########################
        imgContours2 = img.copy()
        biggestContour = rectanglePoints[0]
        peri = cv2.arcLength(biggestContour, True)
        approx = cv2.approxPolyDP(biggestContour, 0.02*peri, True)
        cv2.drawContours(imgContours2, approx, -1, color, 8)
        cv2.imshow("contours2_"+filename, imgContours2)

        #--------------------------------------------------
        imgFixPosition = img.copy()
        imgFixPosition, fixPosition = Utils.FixCornerPositions(imgFixPosition, approx)


        ##################################################################
        fixPosition = Utils.findPositionCorner(approx)
        pts1 = np.float32(fixPosition)
        pts2 = np.float32([[0,0],[widthKTP,0],[0,heightKTP],[widthKTP,heightKTP]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthKTP, heightKTP))
        cv2.imshow("wrap_"+filename, imgWarpColored)

        ####################################################
        #name
        cv2.imwrite(resultImagesPath+"V2_"+filename[7:], imgWarpColored)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':       
    # Calling main() function 
    main()