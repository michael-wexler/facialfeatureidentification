import streamlit as st
import cv2
from PIL import Image
import numpy as np

smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect(image):
    image = np.array(image.convert('RGB'))
    faces = face_classifier.detectMultiScale(image = image, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img = image, pt1 = (x, y), pt2=(x + w, y + h), color = (255, 0, 0), thickness = 2)
        region = image[y : y + h, x : x + w]
        eyes = eye_classifier.detectMultiScale(region)
        smile = smile_classifier.detectMultiScale(region, minNeighbors = 25)
        for (x, y, w, h) in eyes:
            cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in smile:
            cv2.rectangle(region, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image, faces

def main():
    st.markdown("<h2 class='title white'>Facial Feature Detection</h2>", unsafe_allow_html = True)
    st.markdown("<div class='title blue'>This facial feature detection web application detects faces with red rectangles, eyes with green rectangles, and smiles with blue rectangles. </div>", unsafe_allow_html = True)
    image_file = st.file_uploader("Upload image", type = ['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:
    	image = Image.open(image_file)
    	if st.button("Detect Facial Features"):
    		result_img, result_faces = detect(image = image)
    		st.image(result_img, use_column_width = True)
    		st.success("Found {} faces\n".format(len(result_faces)))

if __name__ == "__main__":
    main()
