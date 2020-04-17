import cv2
import numpy as np

# create object of CascadeClassifier that is face_classifier.
face_classifier = cv2.CascadeClassifier('/home/anjali/anaconda3/pkgs/libopencv-3.4.2-hb342d67_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

#Function to extract the features of face.
def face_extractor(imag):

    gray = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = imag[y:y+h,x:x+w]
        return cropped_face


cap = cv2.VideoCapture(0)
count =0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = '/home/anjali/Pictures/OpenCV-master/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==100: #ASCII COde for enter is 13
        break

cap.release()
cv2.destroyAllWindows()
print("Collection sample complete")



