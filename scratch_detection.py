from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from matplotlib import pyplot as plt

isWheel = True

count = 1

def framework(img, coords, accuracy):
    plt.figure()

    L=[]
    arr=[]
    
    model = VGG16()
    imgi=0
    imgj=0
    inc = int(max(img.shape[0], img.shape[1])/2)
    while imgi<=img.shape[0]:
        while imgj<=img.shape[1]:
            croppedimg=img[imgi:min(imgi+inc,img.shape[0]),imgj:min(imgj+inc, img.shape[1])]
            if abs(imgi - min(imgi+inc,img.shape[0])) > 10 and abs(imgj - min(imgj+inc, img.shape[1])) >= 10:
                crimg=cv2.resize(croppedimg, (224, 224))
                crimg = img_to_array(crimg)
                crimg = crimg.reshape((1,crimg.shape[0], crimg.shape[1], crimg.shape[2]))
                crimg = preprocess_input(crimg)
                pred = model.predict(crimg)
                label = decode_predictions(pred)
                label = label[0][0]
                
                if 'wheel' in label[1]:
                    coords.append([[imgi,min(imgi+inc,img.shape[0])],[imgj,min(imgj+inc, img.shape[1])]])
                    accuracy.append(float(label[2]))
                    arr.append(croppedimg)

                if float(label[2]*100) > float(10):
                    L.append(label[1])
            imgj+=inc
        imgi+=inc
        imgj=0
        
    return(arr)
def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    r1, r2 = sorted(contours, key=cv2.contourArea)[-3:-1]
    x, y, w, h = cv2.boundingRect(np.r_[r1, r2])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

img = cv2.imread(r"car.jpg")
cv2.imshow('Original', img)
cv2.waitKey(0)
coords = []
accuracy = []
if len(framework(img, coords, accuracy)) == 0:
    isWheel = False
else:
    framework(framework(img, coords, accuracy)[0], coords, accuracy)
    arr = coords[accuracy.index(max(accuracy))]
    print(arr)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gSize = 31
gray = cv2.GaussianBlur(gray_img,(gSize,gSize), 0)
cv2.imshow('Gaussian Blur', gray)
cv2.waitKey(0)

laplacian = cv2.Laplacian(gray,cv2.CV_64F)
cv2.imshow('Laplacian', gray)
cv2.waitKey(0)

binr = cv2.threshold(laplacian, 0,255, cv2.THRESH_BINARY)[1]
#im = cv2.threshold(im, 175 , 250, cv2.THRESH_BINARY)
k = 2
kernel = np.ones((k,k), np.uint8)
closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)

kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,kernel, iterations=1)

lines_list =[]
opening = img_to_array(opening, dtype='uint8')
lines = cv2.HoughLinesP(opening, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=10 # Max allowed gap between line for joining them
            )
cv2.imshow('Hough Lines', opening)
cv2.waitKey(0)

if len(lines_list) > 0:
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv2.line(gray_img,(x1,y1),(x2,y2),(0,255,0),2)
        lines_list.append([(x1,y1),(x2,y2)])

final_output = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)[1]

M = []
for i in range(len(final_output)):
    L = final_output[i].tolist()
    M.append(L)

for i in range(len(M)):
    if(not isWheel):
        for j in range(len(M[i])):
            if M[i][j] == 255:
                img[i][j] = 0
    else:
        for j in range(len(M[i])):
            if M[i][j] == 255 and not(arr[0][0] < j < arr[0][1]) and  not(arr[1][0] < i < arr[1][1]):
                img[i][j] = 0

cv2.imshow('Final Output', img)
cv2.waitKey(0)