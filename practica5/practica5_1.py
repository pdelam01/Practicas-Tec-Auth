'''
Practica 5: Extracción de Rostros/Reconocimiento Facial
    - Separa los rostros encontrados en una imagen
    - Almacena los rostros encontrados en una carpeta llamada "Rostros"
'''

import cv2
import os
import numpy as np

#Cargamos el clasificador
clasificador = cv2.CascadeClassifier('./practica4/haarcascade_frontalface_default.xml')

#Cargamos la imagen a analizar
image = cv2.imread('./practica4/oficina_caras.jpg')

#Transformamos la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Usamos una imagen auxiliar para evitar el recuadro, si no lo hacemos asi, también se almacenara el contorno en la imagen de la cara
imageAux = image.copy()
faces = clasificador.detectMultiScale(gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    maxSize=(200,200))

contador = 0
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    #Recortamos el rostro de la imagen de entrada
    rostro = imageAux[y:y+h, x:x+w]

    #Redimensionamos los rostros detectado.
    rostro= cv2.resize(rostro,(150,150),
    interpolation=cv2.INTER_CUBIC)

    #Guardamos las caras
    cv2.imwrite('./practica5/rostros/rostro_{}.jpg'.format(contador),rostro)

    contador = contador +1

cv2.imshow('rostro', rostro)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
