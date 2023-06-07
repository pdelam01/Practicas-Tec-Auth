'''
Practica 1: Obtención y Segmentación de imágenes
    - Cargar una imagen en escala de grises
    - Guardar una imagen en escala de grises
    - Visualizar una imagen
'''

import cv2

# Cargamos la imagen
img = cv2.imread('imagen.jpg',0)

# La función imwrite va a almacenar una variable de tipo imagen en una
cv2.imwrite('/practica1/grises.jpg', img)

# imshow se utilizara para visualizar las imagenes
cv2.imshow('Visualizando imagen', img)

# WaitKey se utiliza para cerrar las visualizaciones, podemos
# establecer un tiempo, que se cierre al pulsar una tecla concreta o
# cualquier tecla.
cv2.waitKey(10000)

# Destruye todas las ventanas creadas
cv2.destroyAllWindows()
