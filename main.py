import streamlit as st
import numpy as np
import cv2
from PIL import Image

smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def identify(image):
    image = np.array(image.convert('RGB'))
    faces = face_classifier.detectMultiScale(image = image, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img = image, pt1 = (x, y), pt2=(x + w, y + h), color = (153, 50, 204), thickness = 2)
        region = image[y : y + h, x : x + w]
        eyes = eye_classifier.detectMultiScale(region)
        smile = smile_classifier.detectMultiScale(region, minNeighbors = 25)
        for (x, y, w, h) in smile:
            cv2.rectangle(region, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (x, y, w, h) in eyes:
            cv2.rectangle(region, (x, y), (x + w, y + h), (137, 207, 240), 2)
    return image, faces

def main():
    st.markdown("<h2>Facial Feature Identification</h2>", unsafe_allow_html = True)
    st.markdown("<div>This facial feature identification web application identifies faces with purple "
                "rectangles, eyes with blue rectangles, and smiles with red rectangles. </div>", unsafe_allow_html
    = True)
    upload_image = st.file_uploader("Upload image", type = ['jpeg', 'png', 'jpg', 'webp'])
    if upload_image is not None:
    	image = Image.open(upload_image)
    	if st.button("Identify Facial Features"):
    		img, all_faces = identify(image = image)
    		st.image(img, use_column_width = True)
    		st.success("{} faces have been identified\n".format(len(all_faces)))

if __name__ == "__main__":
    main()
