'''
Practica 4: Aprendizaje Supervisado y Detección de Rostros
    - Aprendizaje Supervisado
    - Detección de Rostros en imágenes
'''

import cv2
import numpy as np

# Cargamos el clasificador
clasificador = cv2.CascadeClassifier('practica4/haarcascade_frontalface_default.xml')
image = cv2.imread('practica4/oficina_caras.jpg')

# Transformamos la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Funciona tanto en color como en escala de grises

# Llamamos al clasificador, pasandole la imagen en escala de grises

# scaleFactor establece cuanto se escala la imagen. Esta configuración reduce la imagen en un 10%. Si la imagen se escala
# demasiado se pierde información y no reconocerá todas las caras

# Si la imagen se escala poco, se van a usar mas cantidad de imagen aumentando el tiempo y dando falsos positivos.
# Se aplica una pirámide de imágenes ya que unos rostros pueden ocupar mas o menos en la imagen y es necesario para capturar la información
faces = clasificador.detectMultiScale(gray,
    scaleFactor=1.1,
    minNeighbors=5, # los n vecinos son los cuadros delimitadores de un rostro
    minSize=(30,30), #Tamaño mínimo del objeto, los objetos mas pequeños son ignorados
    maxSize=(200,200))

# Si se detecta algún rostro con estos parámetros del clasificador se almacenan a continuación para ponerles contorno
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()