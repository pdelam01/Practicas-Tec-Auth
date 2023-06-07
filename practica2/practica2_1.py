'''
Practica 2: Deteccion de colores
    - Detectar colores en video tiempo real
    - Detectar colores en tiempo real con contornos
    - Detectar colores en tiempo real con contornos y suavizado
    - Aplicar mascara de color en tiempo real
'''

import cv2
import numpy as np

class Colores():
    """docstring for Video"""
    def __init__(self):
        pass

    #Leemos las imagenes de la camara
    def detectarColor(self):

        #Estos son los rangos para cada una de las componentes, 4 para los bajos y altos de rojos. 
        # Si fuera ej. el color azul, solo serían 2
        redBajo1 = np.array([0,100,20],np.uint8)
        redAlto1 = np.array([8,255,255],np.uint8)
        redBajo2 = np.array([175,100,20],np.uint8)
        redAlto2 = np.array([179,255,255],np.uint8)

        #Detectamos el color azul
        blueBajo = np.array([100,100,20],np.uint8)
        blueAlto = np.array([125,255,255],np.uint8)

        captura = cv2.VideoCapture(0)
        while True:
            ret,frame=captura.read()

            if ret == True:
                #Transformamos los colores de RGB a HSV
                frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                
                #Determinamos los Rangos del color que queremos detectar desde redbajo1 a redalto1 
                # e igual con los altos, asi detecta todo espacio de rojo
                maskRed1 = cv2.inRange(frameHSV, redBajo1,redAlto1)
                maskRed2 = cv2.inRange(frameHSV, redBajo2,redAlto2)
                maskRed = cv2.add(maskRed1, maskRed2)

                #Detectamos el color azul
                maskBlue = cv2.inRange(frameHSV, blueBajo,blueAlto)

                # ============================================================================= 1º parte: detectar los Rojos
                #Ahora vamos a mostrar en la imagen el color de la mascara

                #Visualizamos los colores RED
                #cv2.imshow('maskRed',maskRed)
                #cv2.imshow('frame', frame)

                # ============================================================================= 2º parte: detectar los Azules
                #Visualizamos los colores BLUE
                #cv2.imshow('maskBlue',maskBlue)
                #cv2.imshow('frame', frame)

                # ============================================================================= 3º parte: cambiar los Blancos por la mascara
                #Mostramos la mascara de color por pantalla en vez de blanco en el color detectado
                #res = cv2.bitwise_and(frame,frame, mask= maskBlue)
                #cv2.imshow('frame',frame)
                #cv2.imshow('maskRed',maskRed)
                #cv2.imshow('res',res)

                # ============================================================================= 4º parte: mostrar los contornos
                #Mostramos el contorno
                _,contorno,_ = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #Dibujamos los contornos, el -1 significa que dibuja todos los contornos
                #(255,0,0) es el color en el que vamos a pintar los contornos en RGB, 3 es el
                #grosor de la linea
                for c in contorno:
                    area=cv2.contourArea(c)
                    if area >3000:
                        # ===================================================================== 5º parte: suavizamos los contornos
                        hull = cv2.convexHull(c)
                        cv2.drawContours(frame, [hull], 0, (255,0,0), 3)
                        cv2.imshow('frame', frame)
                
                # Se cierra cuando pulsamos la s
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

        captura.release()
        
# Instanciamos la clase Colores
colores = Colores()

# Llamamos a la funcion detectarColor
colores.detectarColor()

cv2.destroyAllWindows()
