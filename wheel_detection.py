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