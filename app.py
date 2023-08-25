import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
import pyttsx3

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h', 'Speed Limit 70 km/h',
        'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h', 'No passing',
        'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection', 'Priority road', 'Yield',
        'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
        'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
        'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo]


def main():
    frameWidth = 640  # CAMERA RESOLUTION
    frameHeight = 480
    brightness = 180
    threshold = 0.75
    font = cv2.FONT_HERSHEY_SIMPLEX

    model = load_model('traffic_m3.h5')

    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)

    st.title("Traffic Sign Recognition")

    stframe = st.empty()

    while True:
        success, imgOrignal = cap.read()

        if not success:
            continue

        img = np.asarray(imgOrignal)

        try:
            img = cv2.resize(img, (32, 32))
        except cv2.error:
            print("Error: Unable to resize the image.")
            break

        img = preprocessing(img)

        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)
        if probabilityValue > threshold:
            cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)
            text=str(getClassName(classIndex))
            text_to_speech(text)
        img_rgb = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB")

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
