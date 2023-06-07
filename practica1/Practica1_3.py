'''
Practica 1: Obtención y Segmentación de imágenes
    - Binarizar una imagen
    - Crear una imagen en escala de grises y binarizarla
    - Trabajar distntos tipos de binarización
'''

import cv2
import numpy as np

# ============================================================================= 1º parte: separar objeto de fondo

#Leemos la imagen afilapuntas.jpg en grises
afilapuntas = cv2.imread('practica1/afilapuntas.jpg',0)

#Mostramos la imagen afilapuntas.jpg
cv2.imshow('Afilapuntas',afilapuntas)

#Sacamos el umbral de la imagen afilapuntas.jpg
_,binarizada = cv2.threshold(afilapuntas,150,255,cv2.THRESH_BINARY)

#Mostramos la imagen afilapuntas.jpg binarizada
cv2.imshow('Afilapuntas binarizada',binarizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Tipos binarización:
#   o Binaria: THRESH_BINARY
#   o Binaria invertida: THRESH_BINARY_INV
#   o Truncada: THRESH_TRUNC
#   o Truncada invertida: THRESH_TOZERO


# ============================================================================= 2º parte: creamos imagen en grises y la binarizamos

#Creamos una imagen de 500x600 en escala de grises
grises = np.zeros((500,600),dtype=np.uint8)

#Tipo de fuente a utilizar
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(grises,'Umbral: T=130',(100,70), font,
1.5,(255),2,cv2.LINE_AA)

#Se divide a la imagen en secciones, para luego a cada una de ellas asignarle grises diferentes.
grises[100:300,:200] = 130
grises[100:300,200:400] = 20
grises[100:300,400:600] = 210
grises[300:600,:200] = 35
grises[300:600,200:400] = 255
grises[300:600,400:600] = 70

#Hacemos una copia
grises2=grises.copy()

#Escribimos el texto en las imagenes
cv2.putText(grises,'130',(60,150), font, 1, (255), 1, cv2.LINE_AA)
cv2.putText(grises,'20',(280,150), font, 1,(255), 1, cv2.LINE_AA)
cv2.putText(grises,'210',(470,150), font, 1,(0), 1, cv2.LINE_AA)
cv2.putText(grises,'35',(70,350), font, 1, (255), 1, cv2.LINE_AA)
cv2.putText(grises,'255',(270,350), font, 1, (0), 1, cv2.LINE_AA)
cv2.putText(grises,'70',(480,350), font, 1, (255), 1, cv2.LINE_AA)
cv2.putText(grises,'130>T?',(40,230), font, 1, (255), 1, cv2.LINE_AA)
cv2.putText(grises,'20>T?',(250,230), font, 1, (255), 1, cv2.LINE_AA)
cv2.putText(grises,'210>T?',(440,230), font, 1, (0), 1, cv2.LINE_AA)
cv2.putText(grises,'35>T?',(50,430), font, 1, (255), 1, cv2.LINE_AA)
cv2.putText(grises,'255>T?',(240,430), font, 1, (0), 1, cv2.LINE_AA)
cv2.putText(grises,'70>T?',(450,430), font, 1, (255), 1, cv2.LINE_AA)

#130 es el umbral, solo se muestran las imágenes entre 130 y 255
_,binarizada = cv2.threshold(grises2,130,255,cv2.THRESH_BINARY)
cv2.imshow('Grises',grises)
cv2.imshow('Grises2',binarizada)
cv2.waitKey(0)
cv2.destroyAllWindows() 