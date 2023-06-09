﻿# Prácticas de Tecnologías de Autenticación

Este repositorio contiene las prácticas realizadas en la asignatura de Tecnologías de Autenticación del Máster en Investigación en Ciberseguridad. A continuación, se proporciona una descripción de cada práctica:

## Práctica 1: Obtención y Segmentación de imágenes

    - Cargar una imagen en escala de grises
    - Guardar una imagen en escala de grises
    - Visualizar una imagen

## Práctica 2: Detección de colores

    - Detectar colores en video tiempo real
    - Detectar colores en tiempo real con contornos
    - Detectar colores en tiempo real con contornos y suavizado
    - Aplicar máscara de color en tiempo real

## Práctica 3: Suma de imágenes y Detección de movimiento

    - Para detectar movimiento tendremos que seguir los siguientes pasos:
    - Leer un video o realizar video streaming
    - Transformar de BGR a escala de grises
    - Conseguir la imagen del fondo y exterior, para restarlas con cv2.absdiff
    - Aplicar umbralización simple
    - Encontrar los contornos
    - Discriminar los contornos encontrados de acuerdo a su tamaño y encerrar en un rectángulo a los que superen cierta área

## Práctica 4: Aprendizaje Supervisado y Detección de Rostros

    - Aprendizaje supervisado
    - Detección de rostros en video en streaming
    - Recuadro en verde para frontal y en azul para perfil

## Práctica 5: Extracción de Rostros/Reconocimiento Facial

    - Encuentra rostros en video en streaming
    - Almacena los rostros encontrados en una carpeta al pulsar "s"

## Práctica final: Proyecto de reconocimiento facial con OpenCV

    - La práctica final es un proyecto de reconocimiento facial de usuarios registrados utilizando la biblioteca OpenCV. 
    - El proyecto permite identificar a los usuarios registrados y muestra su nombre como identificación. 
    - Si se detecta a un desconocido, se marcará como tal. 
    - Además, el proyecto incluye la detección de imágenes estáticas para prevenir ataques mediante la detección de parpadeo.
