import cv2
import numpy as np
import face_recognition

imgTom = face_recognition.load_image_file('ImagesBasic/Tom Holland.jpg')
imgTom = cv2.cvtColor(imgTom, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Tom Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#imgTom = face_recognition.load_image_file('ImagesBasic/Gustavo Cestero.JPG')
#imgTom = cv2.cvtColor(imgTom, cv2.COLOR_BGR2RGB)
#imgTest = face_recognition.load_image_file('ImagesBasic/Cestero Test1.JPG')
#imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgTom)[0]
encodeTom = face_recognition.face_encodings(imgTom)[0]
cv2.rectangle(imgTom,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]), (255,0,255), 2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encodeTom],encodeTest)
faceDistance = face_recognition.face_distance([encodeTom],encodeTest)
print(results, faceDistance)
cv2.putText(imgTest,f'{results} {round(faceDistance[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Tom Holland', imgTom)
cv2.imshow('Tom Test', imgTest)
cv2.waitKey(0)
