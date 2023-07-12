import cv2
import numpy as np
import sys
import os
import face_recognition as face
import speech_recognition as sr
from pymongo import MongoClient
import threading
import time

if os.path.exists('camara.py'):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()


mostraridiomas = True
mostrargrado = True
mostrarcontacto = True
mostraraptitudes = True
final = False
ocupado = False
reconoce_rostro = True

#Función para reconocer el audio mediante Google
def RecognizeAudio():
    rec = sr.Recognizer()
    mic = sr.Microphone(device_index = 0)
    with mic as source:
        try:
            #rec.adjust_for_ambient_noise(source, duration=0.5)
            audio = rec.listen(source)
            palabra = rec.recognize_google(audio, language="es-ES")            
            return palabra
        except sr.UnknownValueError:
            print("No se pudo reconocer el audio. Por favor, intenta nuevamente.")
            return None

def ReconocimientoPalabras():
    global final
    global mostraridiomas
    global mostrargrado
    global mostrarcontacto
    global mostraraptitudes
    global ocupado
    while not final:
        print("Di 'Alexa' para empezar")
        palabra_alexa = RecognizeAudio()
        print(palabra_alexa)
        if(palabra_alexa == 'Alexa'):
            print("Di 'idioma', 'grado', 'contacto' o 'aptitudes' para ocultar 'salir' para terminar la ejecucción")
            palabra = RecognizeAudio()
            if(palabra == 'idioma'):
                print(palabra)
                mostraridiomas = not mostraridiomas
            elif(palabra == 'grado'):
                print(palabra)
                mostrargrado = not mostrargrado
            elif(palabra == 'contacto'):
                print(palabra)
                mostrarcontacto = not mostrarcontacto
            elif(palabra == 'aptitudes'):
                print(palabra)
                mostraraptitudes = not mostraraptitudes
            elif(palabra == 'salir'):
                final = True


def ReconocimientoFacial():
    global final
    global mostraridiomas
    global mostrargrado
    global mostrarcontacto
    global mostraraptitudes
    global ocupado
    global reconoce_rostro
    DIC = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parametros = cv2.aruco.DetectorParameters()

    client = MongoClient(host=['localhost:27017'])
    db = client.autentication
    cap = cv2.VideoCapture(0)


    if cap.isOpened():
        hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Tamaño del frame de la cámara: ", wframe, "x", hframe)

        matrix, roi = cv2.getOptimalNewCameraMatrix(camara.cameraMatrix, camara.distCoeffs, (wframe,hframe), 1, (wframe,hframe))
        roi_x, roi_y, roi_w, roi_h = roi 
        while not final:
            ret, framebgr = cap.read()

            #Rectificamos la imagen
            framerectificado = cv2.undistort(framebgr, camara.cameraMatrix, camara.distCoeffs, None, matrix)
            framerecortado = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            #Utilizamos el gris para distinguir mejor las esquinas
            gray = cv2.cvtColor(framebgr,cv2.COLOR_BGR2GRAY)
            framerectificado_gray = cv2.undistort(gray, camara.cameraMatrix, camara.distCoeffs, None, matrix)
            framerecortado_gray = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if ret:     
                # Aquí procesamos el frame
                (corners, ids, rejected) = cv2.aruco.detectMarkers(framerecortado_gray, DIC, parameters=parametros)
                cv2.imshow("Tarjeta de contacto Facial", framerecortado)

                #Introducimos las imagenes que tenemos en la base de datos
                imagenes = []
                for img in db.users.find():
                    imagenes.append(str(img["imagen"]))

                for img in imagenes:
                    user = cv2.imread("imgs/"+img) 
                    userrgb = cv2.cvtColor(user, cv2.COLOR_BGR2RGB) 
                    locs_user = face.face_locations(userrgb)
                    cod_user = face.face_encodings(userrgb, locs_user, model='small')[0]

                    miniframe = cv2.resize(cv2.cvtColor(framerecortado, cv2.COLOR_BGR2RGB), None, fx=0.25, fy=0.25)
                    locs = face.face_locations(miniframe)
                    cods = face.face_encodings(miniframe, locs, model='small')
                    frame_copia = framerecortado

                    #Si se detecta algun rostro
                    if locs is not None:

                        for i in range(len(locs)):

                            #Comprobamos si concuerda con algún rostro
                            if face.compare_faces(cod_user, [cods[i]], 0.7)[0]:
                                
                                t, r, b, l = locs[i]
                                dist = face.face_distance(cod_user, [cods[i]])[0]

                                usuario = db.users.find_one( {"imagen": img})

                                imagen = cv2.imread("imgs/"+img) 

                                nombre = usuario['name']
                                nombre_size, _ = cv2.getTextSize(nombre, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                                grado = usuario['grado']
                                grado_size, _ = cv2.getTextSize(grado, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)

                                idiomas_1 = usuario['idiomas_1']
                                idiomas_2 = usuario['idiomas_2']
                                idiomas_3 = usuario['idiomas_3']

                                aptitudes_1 = usuario['aptitudes_1']
                                aptitudes_2 = usuario['aptitudes_2']
                                aptitudes_3 = usuario['aptitudes_3']

                                gmail = usuario['gmail']
                                github = usuario['github']
                                telefono = usuario['telefono']
                                linkedin = usuario['linkedin']

                                #Calculamos a posición del texto
                                centro_x = int((l*4+ r*4)/2)
                                centro_y = int((t*4+ b*4)/2)

                                #La posición del nombre
                                nombre_posicion_x = centro_x - int(nombre_size[0] / 2)
                                nombre_posicion_y = centro_y - int(frame_copia.shape[1] / 4)
                                
                                #La posición del grado
                                grado_posicion_x = centro_x - int(grado_size[0] / 2)
                                grado_posicion_y = centro_y + int(frame_copia.shape[1] / 4)

                                #La posición del idioma
                                idioma_posicion_x = centro_x + 150
                                idioma_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 40

                                #La posición de aptitudes
                                ap_posicion_x = centro_x + 150
                                ap_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 140

                                #La posición de la información de git
                                git_posicion_x = centro_x - 270
                                git_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 60

                                #La posición de la información de gmail
                                gmail_posicion_x = centro_x - 270
                                gmail_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 90

                                #La posición de la información del telefono
                                tlf_posicion_x = centro_x - 270
                                tlf_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 130

                                #La posición de la información del linkedin
                                linkedin_posicion_x = centro_x - 270
                                linkedin_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 170

                                #La posición del logo github
                                logogit_posicion_x = centro_x - 310
                                logogit_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 40

                                #La posición del logo gmail
                                logogmail_posicion_x = centro_x - 310
                                logogmail_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 80

                                #La posición del logo telefono
                                logotlf_posicion_x = centro_x - 310
                                logotlf_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 110

                                #La posición del logo linkedin
                                logolinkedin_posicion_x = centro_x - 310
                                logolinkedin_posicion_y = centro_y - int(frame_copia.shape[1] / 4) + 150

                                #Mostramos el texto y los iconos                        
                                cv2.putText(frame_copia, nombre, (nombre_posicion_x,nombre_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)                      

                                if(mostrargrado):
                                    cv2.putText(frame_copia, grado, (grado_posicion_x,grado_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)

                                if(mostraridiomas):
                                    cv2.putText(frame_copia, idiomas_1, (idioma_posicion_x,idioma_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, idiomas_2, (idioma_posicion_x,idioma_posicion_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, idiomas_3, (idioma_posicion_x,idioma_posicion_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                
                                if(mostraraptitudes):
                                    cv2.putText(frame_copia, aptitudes_1, (ap_posicion_x,ap_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, aptitudes_2, (ap_posicion_x,ap_posicion_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, aptitudes_3, (ap_posicion_x,ap_posicion_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                                if(mostrarcontacto):
                                    cv2.putText(frame_copia, github, (git_posicion_x,git_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, gmail, (gmail_posicion_x,gmail_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, telefono, (tlf_posicion_x,tlf_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame_copia, linkedin, (linkedin_posicion_x,linkedin_posicion_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

                                    #Superponer logo git
                                    github_logo_original = cv2.imread("imgs/github_logo.png")
                                    github_logo = cv2.resize(github_logo_original,(30,30), interpolation=cv2.INTER_CUBIC)

                                    if (logogit_posicion_y > 0 and logogit_posicion_x > 0):
                                        frame_copia[logogit_posicion_y:logogit_posicion_y+github_logo.shape[0], logogit_posicion_x:logogit_posicion_x+github_logo.shape[1]] = github_logo

                                    #Superponer logo gmail
                                    gmail_logo_original = cv2.imread("imgs/gmail_logo.png")
                                    gmail_logo = cv2.resize(gmail_logo_original,(30,20), interpolation=cv2.INTER_CUBIC)

                                    if (logogmail_posicion_y > 0 and logogmail_posicion_x > 0):
                                        frame_copia[logogmail_posicion_y:logogmail_posicion_y+gmail_logo.shape[0], logogmail_posicion_x:logogmail_posicion_x+gmail_logo.shape[1]] = gmail_logo

                                    #Superponer logo telefono
                                    tlf_logo_original = cv2.imread("imgs/telefono_logo.png")
                                    tlf_logo = cv2.resize(tlf_logo_original,(30,30), interpolation=cv2.INTER_CUBIC)

                                    if (logotlf_posicion_y > 0 and logotlf_posicion_x > 0):
                                        frame_copia[logotlf_posicion_y:logotlf_posicion_y+tlf_logo.shape[0], logotlf_posicion_x:logotlf_posicion_x+tlf_logo.shape[1]] = tlf_logo

                                    #Superponer logo linkedin
                                    linkedin_logo_original = cv2.imread("imgs/linkedin_logo.png")
                                    linkedin_logo = cv2.resize(linkedin_logo_original,(30,30), interpolation=cv2.INTER_CUBIC)

                                    if (logolinkedin_posicion_y > 0 and logolinkedin_posicion_x > 0):
                                        frame_copia[logolinkedin_posicion_y:logolinkedin_posicion_y+linkedin_logo.shape[0], logolinkedin_posicion_x:logolinkedin_posicion_x+linkedin_logo.shape[1]] = linkedin_logo                   
                                    
                                    reconoce_rostro = True
                            else:
                                reconoce_rostro = False

                    cv2.imshow("Tarjeta de contacto Facial", framerecortado)

                if(not reconoce_rostro):
                    ocupado = True
                    archivo_imagen = framerecortado
                    nombre = input('Nombre completo: ')

                    imagen = input('Nombre de la imagen: ')
                
                    grado = input('Grado: ')

                    idiomas_1 = input('Idioma 1 y su nivel: ')
                    idiomas_2 = input('Idioma 2 y su nivel: ')
                    idiomas_3 = input('Idioma 3 y su nivel: ')

                    aptitudes_1 = input('Aptitudes 1: ')
                    aptitudes_2 = input('Aptitudes 2: ')
                    aptitudes_3 = input('Aptitudes 3: ')

                    gmail = input('Direccion de gmail: ')
                    github = input('Usuario de github: ')
                    telefono = input('Numero de telefono: ')
                    linkedin = input('Usuario de Linkedin: ')

                    usuario_nuevo = {"name": nombre, "imagen":imagen+".jpg", "grado":grado, "idiomas_1": idiomas_1, "idiomas_2": idiomas_2, "idiomas_3": idiomas_3, "aptitudes_1": aptitudes_1, "aptitudes_2": aptitudes_2, "aptitudes_3": aptitudes_3, "gmail": gmail, "github": github, "telefono": telefono, "linkedin": linkedin}

                    cv2.imwrite("imgs/"+imagen+".jpg", archivo_imagen)
                    
                    db.users.insert_one(usuario_nuevo)

                    ocupado = False
                if cv2.waitKey(1) == ord(' '):
                    final = True
            else:
                final = True
    else:
        print("No se pudo acceder a la cámara.")



t1 = threading.Thread(target=ReconocimientoPalabras)
t2 = threading.Thread(target=ReconocimientoFacial)

t1.start()
t2.start()