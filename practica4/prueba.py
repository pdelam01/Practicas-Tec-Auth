import cv2
import numpy as np

# Cargamos los clasificadores
clasificador_frontal = cv2.CascadeClassifier('practica4/haarcascade_frontalface_default.xml')
clasificador_perfil = cv2.CascadeClassifier('practica4/haarcascade_profileface.xml')

video = cv2.VideoCapture(0)

while video.isOpened():
    ret, imagen = video.read()
    if ret:
        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Detectamos rostros frontales
        faces_frontal = clasificador_frontal.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            maxSize=(200,200))

        # Detectamos rostros laterales
        faces_perfil = clasificador_perfil.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            maxSize=(200,200))

        # Convertimos las matrices de coordenadas a una lista de tuplas
        faces_frontal = [tuple(face) for face in faces_frontal]
        faces_perfil = [tuple(face) for face in faces_perfil]

        # Unimos ambas listas de rostros detectados
        faces = faces_frontal + faces_perfil

        # Dibujamos un único rectángulo alrededor de todos los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 150), 2)
            print(faces)

        cv2.imshow('video', imagen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()