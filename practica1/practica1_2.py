'''
Practica 1: Obtención y Segmentación de imágenes
    - Capturar video con cámara en tiempo real
    - Leer video capturado y/o descargado
'''

import cv2

# Definimos clase Video
class Video:

    def __init__(self):
        pass

    def capturarVideo(self):
        # Capturamos el video, si es 0 es la webcam del ordenador y si es 1 es una camara externa
        captura = cv2.VideoCapture(0)

        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Guardamos el video capturado en una variable
        # Los parametros de la funcion VideoCapture son: nombre, codec, fps, tamaño
        salida = cv2.VideoWriter('videoCapturado.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

        while(captura.isOpened()):
            # Leemos el video
            ret, imagen = captura.read()
            if ret == True:
                # Mostramos el video
                cv2.imshow('video', imagen)
                # Guardamos el video
                salida.write(imagen)
                # Si pulsamos la tecla s se guardara la imagen
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            else:
                break
        
        # Liberamos la camara y cerramos todas las ventanas
        captura.release()
        salida.release()
        cv2.destroyAllWindows()
    
    def leerVideo(self):
        print('Leyendo video...')
        captura = cv2.VideoCapture('videoCapturado.mp4')
        while(captura.isOpened()):
            ret, imagen = captura.read()
            if ret == True:
                print("Mostrando video")
                cv2.imshow('video', imagen)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            else:
                break
        captura.release()
        cv2.destroyAllWindows()

# Instanciamos la clase Video
video = Video()

# Llamamos a la funcion capturarVideo
#video.capturarVideo()

# Llamamos a la funcion leerVideo
video.leerVideo()