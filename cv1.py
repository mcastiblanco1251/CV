
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
#from pynput.keyboard import Key, Controller


#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im = Image.open('ml.jpg')
st.set_page_config(page_title='CV-App', layout="wide", page_icon=im)
st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    image = Image.open('ml.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    st.write("""
    # Detección Visual
    Esta App utiliza algoritmos de reconocimiento de imagen con interraciones!
    """)
    with st.expander("Contáctanos 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name = st.text_input('Name')
            mail = st.text_input('Email')
            q = st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n' + name + '\n' + mail + '\n' + q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Aplicación')
st.markdown('____________________________________________________________________')
app_des = st.expander('Descripción App')
with app_des:
    st.write("""Esta aplicación muestra detección de personas con algunas interacciones""")

def detect():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect2(l,f):
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))

        if l=='Línea':
            img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 5)
            img = cv2.line(img, (0, height), (width, 0), (0, 255, 0), 5)
            cv2.imshow('frame', img)


        elif l=='Rectángulo':
            img1 = cv2.rectangle(frame, (100, 100), (200, 200), (0, 250, 250), 5)
            cv2.imshow('frame', img1)
        elif l=='Círculo':
            img2 = cv2.circle(frame, (300, 300), 60, (0, 0, 255), -1)
            cv2.imshow('frame', img2)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX

            img3 = cv2.putText(frame, f, (80, height - 50), font, 2, (0, 0, 250), 5, cv2.LINE_AA)
            cv2.imshow('frame', img3)

        if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
def detect3():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        width = int(cap.get(3))
        height = int(cap.get(4))

        image = np.zeros(frame.shape, np.uint8)
        smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180)
        image[height//2:, :width//2] = smaller_frame
        image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180)
        image[height//2:, width//2:] = smaller_frame

        cv2.imshow('frame', image)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def tomar():
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        st.write(type(cv2_img))

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        st.write(cv2_img.shape)
    # if cv2.waitKey(1) == ord('q'):
    #     break
st.warning('Para abrir haga click en el ícono en la parte inferior y Para Detener Presione la letra Q ')

row1_1, row1_2 = st.columns((2, 2))

with row1_1:
    st.subheader('Detección Facial y de Ojos')
    image = Image.open('ml2.jpg')
    st.image(image, use_column_width=False)
    d=st.button('Detectar')
    if d:
        detect()
    # stqq.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
with row1_2:

    st.subheader('Interacción con la imagen')
    titles=['Línea','Círculo', 'Rectángulo', 'Texto']
    l= st.radio("Seleeccione la interacción", titles)#, index=default_radio, key="radio")

    if l=='Texto':
        f='Eres Genial!!!'
    else:
        f=''
    s=st.button('Colocar')
    if s:
        detect2(l,f)

row1_3, row1_4 = st.columns((2, 2))

with row1_3:
    st.subheader('Cambio Posición de la imagen')
    image = Image.open('ml3.jpg')
    st.image(image, use_column_width=False)
    z=st.button('Cambiar')
    if z:
        detect3()

with row1_4:
    st.subheader('Tomar Foto')
    image = Image.open('ml4.jpg')
    st.image(image, use_column_width=False)
    t=st.button('Tomar')
    if t:
        tomar()
st.subheader('Si te interesó y quieres saber como se aplica contactanos en [http://ia.smartecorganic.com.co/contact](http://ia.smartecorganic.com.co/index.php/contact/)')
