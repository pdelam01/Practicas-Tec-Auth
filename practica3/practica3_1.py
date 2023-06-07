'''
Practica 3: Suma de imágenes y Detección de movimiento
    - Suma de imagenes
    - Resta de imagenes
    - Valor absoluto de la resta de imagenes
    - Porcentaje de similitud entre imagenes
'''

import cv2

img1=cv2.imread('practica3/Venus_globe.jpg')
img2=cv2.imread('practica3/Earth_globe.jpg')

# Convertir a escala de grises
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calcular la diferencia absoluta en cada canal
diffR = cv2.absdiff(img1[:,:,0], img2[:,:,0])
diffG = cv2.absdiff(img1[:,:,1], img2[:,:,1])
diffB = cv2.absdiff(img1[:,:,2], img2[:,:,2])

# Sumar las diferencias absolutas
diffSum = cv2.add(diffR, diffG)
diffSum = cv2.add(diffSum, diffB)

# Calcular el porcentaje de similitud
maxDiff = gray1.shape[0] * gray1.shape[1] * 255 * 3 # Valor máximo posible de diferencia
similarity = (maxDiff - diffSum.sum()) / maxDiff * 100

print("Porcentaje de similitud en color: %.2f%%" % similarity)

cv2.imshow("Imagen 1", img1)
cv2.imshow("Imagen 2", img2)
cv2.imshow("Diferencia", diffSum)
cv2.waitKey(0)
cv2.destroyAllWindows()