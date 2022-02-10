import cv2
import numpy as np
import math
import face_recognition

imgTom = face_recognition.load_image_file('ImagesBasic/m.0b0j6v_0002.jpg')
imgTom = cv2.cvtColor(imgTom, cv2.COLOR_BGR2RGB)
#0071, 0103
imgTest = face_recognition.load_image_file('ImagesBasic/m.0b0j6v_0003.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#imgTom = face_recognition.load_image_file('ImagesBasic/Gustavo Cestero.JPG')
#imgTom = cv2.cvtColor(imgTom, cv2.COLOR_BGR2RGB)
#imgTest = face_recognition.load_image_file('ImagesBasic/Cestero Test1.JPG')
#imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#TRAIN IMAGE
faceLocation = face_recognition.face_locations(imgTom)[0]
encodeTom = face_recognition.face_encodings(imgTom)[0]
cv2.rectangle(imgTom,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]), (255,0,255), 2)

#TEST IMAGE
faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]), (255,0,255), 2)

#STEP THREE: Comparing these faces and finding the distance between them

#/////////////////////////////////////////UNDER CONSTRUCTION /////////////////////


def face_distance_to_conf(face_distance, face_match_threshold=0.8):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))



#PERCENTAGE ACCURACY CALCULATOR

if len(encodeTom) == 0 or len(encodeTest)==0:
    print("Encoding error")
else:
    encodeTom
    encodeTom = encodeTom[0]
    encodeTest
    encodeTest = encodeTest[0]
    face_distance = face_recognition.face_distance([encodeTom], encodeTest)
    face_match_percentage = face_distance_to_conf(face_distance)
    face_match_percentage = (np.round(face_match_percentage, 2))[0]
    face_match_percentage = "{:.0%}".format(face_match_percentage)
    if face_distance < 0.6:
        result  = "Match"
    else:
        result = "Not a match"

Dict = {'Results': result,
        'Match Percentage': face_match_percentage}
print(Dict)


#//////////////////////////////////

#HOW similar are the faces... find distance
results = face_recognition.compare_faces([encodeTom],encodeTest)
faceDistance = face_recognition.face_distance([encodeTom],encodeTest)
#print(results, faceDistance)
##print(results, faceDistance, face_distance_to_conf(faceDistance, face_match_threshold=0.6))
print(faceDistance, face_distance_to_conf(faceDistance, 0.6))
cv2.putText(imgTest,f'{results} {round(faceDistance[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results)
print("Results: Not a Match, Match Percentage: 12%")


imS = cv2.resize(imgTom, (640, 480))
imS2 =cv2.resize(imgTest, (640, 480))
cv2.imshow('Tom Holland', imS)
cv2.imshow('Test', imS2)
cv2.waitKey(0)

