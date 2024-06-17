import cv2
import numpy as np
from keras.models import load_model

#model=load_model('C:/Users/HPA02532Y/Documents/Machine Learning/Projects/Facial EMOtion/face_emotions_model1.h5')
model=load_model('C:/Users/HPA02532Y/Documents/Machine Learning/Projects/Facial EMOtion/model_file_30epochs.h5')

faceDetect=cv2.CascadeClassifier('C:/Users/HPA02532Y/Documents/Machine Learning/Projects/Facial EMOtion/haarcascade_frontalface_default.xml') #pre-trained Haar Cascade XML file. This file contains the trained model for detecting frontal faces

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}



frame=cv2.imread('C:/Users/HPA02532Y/Documents/Machine Learning/Projects/Facial EMOtion/faces.jpg')
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces= faceDetect.detectMultiScale(gray, 1.3, 3) #The parameters 1.3 and 3 represent the scale factor and minimum neighbors respectively, which control the accuracy and number of faces detected.
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped=np.reshape(normalize, (1, 48, 48, 1))  #() len(number_of_image), image_height, image_width, channel
    result=model.predict(reshaped)
    label=np.argmax(result, axis=1)[0]
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)  #img, coord1, coord2, color, thickness
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w-25,y),(50,50,255),-1)   #top box for text
    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) #img, text, bottomleftcornercoord, fontFace, fontScale, color, thickness

cv2.imshow("Frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()