import cv2
import numpy as np
from keras.models import load_model
import keras.utils as image
from threading import Thread
import pickle
import pandas as pd
from pandasai import PandasAI
cam = cv2.VideoCapture(0)
face_cade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('model.h5')

isWARNING = False
while True:
    _, img = cam.read()
    face = face_cade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for(x,y,h,w) in face:
        face_img = img[y:y +h, x:x+ w]
        cv2.imwrite('temp.jpg', face_img)
        test_img = image.load_img('temp.jpg', target_size=(64,64))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)


        pred = model.predict(test_img)[0][0]

        if pred ==1:
            cv2.rectangle(img,(x,y),(x+w, y+h),(0,0,255),3 )
            cv2.putText(img,'CO DEO KHAU TRANG', ((x+w)// 2, y+h+20), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w, y+h),(0,0,255),3 )
            cv2.putText(img,'KHONG DEO KHAU TRANG', ((x+w)// 2, y+h+20), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)

        cv2.imshow("image", img)

        key = cv2.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()




            


