'''
Practica 4: Aprendizaje Supervisado y Detección de Rostros
    - Aprendizaje Supervisado
    - Detección de Rostros en video en streaming
    - Recuadro en verde para frontal y en azul para perfil
'''

import cv2
import numpy as np

#Cargamos el clasificador
clasificadorFront = cv2.CascadeClassifier("./Practicas_TecAut/practica4/haarcascade_frontalface_default.xml")
clasificadorSide = cv2.CascadeClassifier("./Practicas_TecAut/practica4/haarcascade_profileface.xml")

#Captamos video desde la webcam
captura = cv2.VideoCapture(0)

# Detectamos rostros en el video en streaming
while(captura.isOpened()):
    # Leemos el video
    ret, imagen = captura.read()
    if ret == True:

        # Transformamos la imagen a escala de grises
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Llamamos al clasificador, pasandole la imagen en escala de grises
        facesFront = clasificadorFront.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5, # los n vecinos son los cuadros delimitadores de un rostro
            minSize=(30,30), #Tamaño mínimo del objeto, los objetos mas pequeños son ignorados
            maxSize=(200,200))
        
        facesSide = clasificadorSide.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=5, # los n vecinos son los cuadros delimitadores de un rostro
            minSize=(30,30), #Tamaño mínimo del objeto, los objetos mas pequeños son ignorados
            maxSize=(200,200))                                  
        
        # Si se detecta algún rostro con estos parámetros del clasificador se almacenan a continuación para ponerles contorno
        for(x,y,w,h) in facesFront:
            cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

        # Si se detecta algún rostro de perfil se le indica contorno azul
        for(x,y,w,h) in facesSide:
            cv2.rectangle(imagen,(x,y),(x+w,y+h),(255,155,0),2)

        # Mostramos el video
        cv2.imshow('video', imagen)

        # Si pulsamos la tecla s se guardara la imagen
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

captura.release()
cv2.destroyAllWindows()