import os
import cv2
import numpy as np
from PIL import Image

os.chdir("C:\chen\School\FYP")

# Load all cascades
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
rightEarCascade = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")
testCascade = cv2.CascadeClassifier("output.xml")

#imageColor = cv2.imread("Beautiful-Hairstyles-For-Round-Face.jpg", cv2.CV_LOAD_IMAGE_COLOR)
imageColor = cv2.imread("c9cb25b39e699bd8e8c5d3a3a947b625.jpg", cv2.CV_LOAD_IMAGE_COLOR)
imageGray = cv2.cvtColor(imageColor, cv2.COLOR_BGR2GRAY)

def getCentre(x, y, width, height):
    xCentre = (x + width)/2
    yCentre = (y + height)/2
    return xCentre, yCentre
    
def drawFaceFeatures(detectedFeatures, xFace, yFace, flag, centrePoints):
    if len(detectedFeatures) != 0:
        for feature in detectedFeatures:
            # Box up the feature
            x1, y1, x2, y2 = feature[0], feature[1], feature[2], feature[3]
            cv2.rectangle(imageColor, (x1 + xFace, y1 + yFace), (x1 + x2 + xFace, y1 + y2 + yFace), cv2.cv.RGB(255, 0, 0), 1)

            # Find centre of the box
            xCentre, yCentre = getCentre((x1 + xFace), (y1 + yFace), (x1 + x2 + xFace), (y1 + y2 + yFace))
            cv2.circle(imageColor, (xCentre, yCentre), 2, (0, 255, 0, 0), -1, 8, 0)
            centrePoints = np.vstack((centrePoints, [xCentre, yCentre, flag]))
    return centrePoints

if __name__ == "__main__":

    centrePoints = [0, 0, 0]

    faces = faceCascade.detectMultiScale(imageGray, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (100, 100))
    if len(faces) != 0:
        for face in faces:
            x, y, width, height = face[0], face[1], face[2], face[3]
            cv2.rectangle(imageColor, (x, y), (x + width, y + height), cv2.cv.RGB(255, 0, 0), 1)
            
            # Draw eyes
            croppedRegion = imageGray[y:((y + height)*0.8), x:((x + width))]
            eyes = eyeCascade.detectMultiScale(croppedRegion, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (100, 100))
            centrePoints = drawFaceFeatures(eyes, x, y, 0, centrePoints)

            # Draw nose
            croppedRegion = imageGray[y:(y + height)*0.9, x:((x + width))]
            noses = noseCascade.detectMultiScale(croppedRegion, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (120, 120))
            centrePoints = drawFaceFeatures(noses, x, y, 1, centrePoints)

            # Draw mouth
            croppedRegion = imageGray[(y*2):((y + height)*0.95), x:((x + width))]
            mouths = mouthCascade.detectMultiScale(croppedRegion, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (100, 100))
            centrePoints = drawFaceFeatures(mouths, x, y*2, 2, centrePoints)
            
            centrePoints = np.delete(centrePoints, 0, 0)

            # assuming detection is correct and no false positives
            eyeToEyeDisance = np.linalg.norm((centrePoints[0, :]) - (centrePoints[1, :]))
            eyeToNoseDistance1 = np.linalg.norm((centrePoints[0, :]) - (centrePoints[2, :]))
            eyeToNoseDistance2 = np.linalg.norm((centrePoints[1, :]) - (centrePoints[2, :]))
            eyeToMouthDistance1 = np.linalg.norm((centrePoints[0, :]) - (centrePoints[3, :]))
            eyeToMouthDistance2 = np.linalg.norm((centrePoints[1, :]) - (centrePoints[3, :]))
            noseToMouthDistance = np.linalg.norm((centrePoints[2, :]) - (centrePoints[3, :]))

    # resize specs
    spectaclesColor = Image.open("Bevel-Spectacles-Oleg_409.png")
    (specsWidth, specsHeight) = spectaclesColor.size
    frameBoxWidth = specsWidth/4
    frameBoxHeight = specsHeight/3
    leftBoxCentre = np.array([frameBoxWidth, frameBoxHeight])
    rightBoxCentre = np.array([frameBoxWidth*3, frameBoxHeight])
    frameEyeToEyeDisance = np.linalg.norm(leftBoxCentre - rightBoxCentre)
    
    factor = eyeToEyeDisance/frameEyeToEyeDisance
    smallSpecsWidth = int(specsWidth * factor)
    smallSpecsHeight = int(specsHeight * factor)
    smallSpecs = spectaclesColor.resize((smallSpecsWidth, smallSpecsHeight))
    smallSpecs.save('smallSpecs.png', "PNG")

    # overlay specs on face
    eyeX1 = centrePoints[0, 0]
    eyeY1 = centrePoints[0, 1]
    eyeX2 = centrePoints[1, 0]
    eyeY2 = centrePoints[1, 1]
    if eyeX1 < eyeX2:
        eyeX = eyeX1
        eyeY = eyeY1
    else:
        eyeX = eyeX2
        eyeY = eyeY2
    print eyeX1
    print eyeY1
    print eyeX2
    print eyeY2
    (width, height) = smallSpecs.size
    x = eyeX - (width/4)
    y = eyeY - (height/3)
    smallSpecs.convert('RGBA')
    background = Image.open("51af9ccd43b8b90711.jpg")
    background.paste(smallSpecs, (x, y), smallSpecs)
    background.save("new2.png","PNG")
    
    #print eyeToEyeDisance
    #print eyeToNoseDistance1
    #print eyeToNoseDistance2
    #print eyeToMouthDistance1
    #print eyeToMouthDistance2
    #print noseToMouthDistance
    cv2.imwrite('detected.jpg', imageColor)
