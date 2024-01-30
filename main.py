from typing import Union
from typing import Annotated

import dlib
from fastapi import FastAPI,  File, UploadFile, Form
import face_recognition
import cv2
import io
import numpy as np


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/face-compare")
async def face_compare(face1: Annotated[str, Form()], face2: Annotated[str, Form()]):
    dlib.DLIB_USE_CUDA = True
    if not face1 or not face2:
        return {'error': 'Two face images are required.'}
    # face2_bytes = face2.file.read()
    # np_array = np.frombuffer(face2_bytes, np.uint8)
    # img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print(fm)
    # if fm < 100: 
    #     return {'error': 'Yuz sifati bilan muommo yuzaga keldi, iltimos kameraga yuzingizni qarating!'}
    # Load face images
    image1 = face_recognition.load_image_file(face1)
    image2 = face_recognition.load_image_file(face2)
    
    encoding1 = face_recognition.face_encodings(image1)
    encoding2 = face_recognition.face_encodings(image2)
    if not encoding2: 
        return {'error': 'No face found FACE2'}
    if not encoding1 :
        return {'error': 'No face found FACE1'}
    
    # Compare faces
    result = face_recognition.compare_faces([encoding1[0]], encoding2[0])
    # result is a list of True/False values indicating if the faces match
    if result[0]:
        return {'match': True, 'confidence': face_recognition.face_distance([encoding1[0]], encoding2[0])[0]}
    else:
        return {'match': False, 'confidence': None}    
    return {"Hello": "World"}