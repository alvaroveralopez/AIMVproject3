import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Variables globales
finalID = None
finalID_backside = None
crop = None
cutID = None
signature = None
rotation_matrix = None
vertex = [None, None, None, None]
cont_frame = 0
control_frame = False
messageID = False
redo = False
angle = 0
rect = None
rectaux = None
xmax, xmin, ymax, ymin = 0, 0, 0, 0
turnedImage = None
heightRectangle, widhtRectangle = 0, 0
capture = cv2.VideoCapture()
template_number = 1
tesseractAvailable = True
webcam = 1
DatosDNI = ""
blursize = 3
areaScreen = 307200


def matchTemplate(dnidetectado):
    global template_number
    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=50)

    # Read the image to compare to
    IDCardTemplate1 = cv2.imread('DNItemplates/DNITemplateFrontal_1.jpg', cv2.IMREAD_GRAYSCALE)
    IDCardTemplate2 = cv2.imread('DNItemplates/DNITemplateFrontal_2.png', cv2.IMREAD_GRAYSCALE)

    # Resize both templates to have 425x270 pixels
    IDCardTemplate1 = cv2.resize(IDCardTemplate1, (425, 270))
    IDCardTemplate2 = cv2.resize(IDCardTemplate2, (425, 270))

    # Detect SIFT features in the ID card templates
    IDCard_keypoints1, IDCard_descriptors1 = sift.detectAndCompute(IDCardTemplate1, None)
    IDCard_keypoints2, IDCard_descriptors2 = sift.detectAndCompute(IDCardTemplate2, None)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(dnidetectado, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features in the frames
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Match the features using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches1 = flann.knnMatch(IDCard_descriptors1, descriptors, k=2)
    matches2 = flann.knnMatch(IDCard_descriptors2, descriptors, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches1 = [m for m, n in matches1 if m.distance < 0.5 * n.distance]
    good_matches2 = [m for m, n in matches2 if m.distance < 0.5 * n.distance]

    # Compare the number of good matches
    template_number = 2 if len(good_matches1) < len(good_matches2) else 1

    return template_number
def detectionID(frame):
    global messageID, redo, cont_frame, control_frame, angle, rect, heightRectangle, widhtRectangle, vertex, areaScreen, blursize
    # Convertir a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"blursize: {blursize}")
    edges = cv2.blur(frame_gray, (blursize, blursize))
    edges = cv2.Canny(edges, 100, 200)

    #kernel = np.ones((5, 5), np.uint8)
    #edges = cv2.dilate(edges, kernel)

    # Encontrar contornos

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Puedes ajustar el color y el grosor

    contours_poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
    boundRect = [cv2.boundingRect(cnt) for cnt in contours_poly]

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i], False)
        if area >= areaScreen/13:
            redo = False
            if not messageID:
                print("DNI detectado, por favor enfocar lo mejor posible.")
            messageID = True

            cv2.rectangle(frame, (boundRect[i][0], boundRect[i][1]),
                          (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), (0, 0, 255), 2)

            cont_frame += 1
            print(f"cont_frame: {cont_frame}")
            if cont_frame >= 30:  # 1 segundo (30) detectando DNI para enfocar bien (30 frames por segundo)
                control_frame = True
                cont_frame = 0
                #print(boundRect[i])
                rect = cv2.minAreaRect(contours[i])
                #print(f"rect: {rect}")
                vertex = cv2.boxPoints(rect)
                vertex = np.intp(vertex)
                angle = rect[2]  # Ángulo del DNI (rectángulo verde)
                heightRectangle = boundRect[i][3]
                widhtRectangle = boundRect[i][2]

        else:
            redo = True
            break

    return frame

def IDrescale(turnedImage):
    global blursize
    vertex_aux = [None, None, None, None]
    xmax, xmin, ymax, ymin = 0, 0, 0, 0

    gray_turnedImage = cv2.cvtColor(turnedImage, cv2.COLOR_BGR2GRAY)
    edgesaux = cv2.blur(gray_turnedImage, (blursize, blursize))
    edgesaux = cv2.Canny(edgesaux, 20, 275)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(edgesaux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
    boundRect = [cv2.boundingRect(cnt) for cnt in contours_poly]


    for i in range(len(contours)):
        area = cv2.contourArea(contours[i], False)
        if area >= areaScreen/13:
            rect = cv2.minAreaRect(contours[i])
            print(f"rect: {rect}")
            vertex_aux = cv2.boxPoints(rect)
            vertex_aux = np.intp(vertex_aux)
            angle = rect[2]  # Ángulo del DNI

            xmax = vertex_aux[0][0]
            xmin = vertex_aux[0][0]
            ymax = vertex_aux[0][1]
            ymin = vertex_aux[0][1]

            # print(f"DNI Reescalado xmax: {xmax}, xmin: {xmin}, ymax: {ymax}, ymin: {ymin}")

    if(xmax==0 and xmin==0 and ymax==0 and ymin==0):
        print("No se ha detectado el DNI correctamente, por favor reinicie el programa.")
        exit()

    for i in range(4):
        if xmax <= vertex_aux[i][0]:
            xmax = vertex_aux[i][0]
        if ymax <= vertex_aux[i][1]:
            ymax = vertex_aux[i][1]
        if xmin >= vertex_aux[i][0]:
            xmin = vertex_aux[i][0]
        if ymin >= vertex_aux[i][1]:
            ymin = vertex_aux[i][1]

    definitivo_aux = turnedImage[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    definitivo_aux = cv2.resize(definitivo_aux, (425, 270))
    return definitivo_aux


def faceExtract(finalID):
    try:
        facecrop = None
        detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        gray = cv2.cvtColor(finalID, cv2.COLOR_BGR2GRAY)
        dest = cv2.equalizeHist(gray)
        rect = detector.detectMultiScale(dest)

        for rc in rect:
            cv2.rectangle(finalID, (rc[0], rc[1]), (rc[0] + rc[2], rc[1] + rc[3]), (0, 0, 0), 0)
            facecrop = finalID[rc[1]:rc[1] + rc[3], rc[0]:rc[0] + rc[2]]

        cv2.imwrite("Imagenes/facecrop.jpg", facecrop)
        cv2.resize(facecrop, (0, 0), fx=6, fy=6)
        cv2.imshow("Face ID", faceExtract(facecrop))
    except:
        print("")

def surnameExtract(finalID):
    global DatosDNI, template_number
    if template_number == 2:
        surname = finalID[80:110, 165:300] # DNI nuevo
    else:
        surname = finalID[80:115, 165:305] # DNI viejo

    surname_gray = cv2.cvtColor(surname, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/surname.jpg", surname_gray)
    cv2.resize(surname_gray, (0, 0), fx=6, fy=6)
    #print("Lectura del surname:")
    #system_output = pytesseract.image_to_string("surname.jpg", config="--oem 3 --psm 6")
    #print(system_output)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(surname_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el apellido")
                DatosDNI += "APELLIDOS: \n"
            else:
                print(f"Apellido: {texto}")
                DatosDNI += f"APELLIDOS: {texto}\n"
        except:
            print("No se ha podido leer el apellido (exception)")

    return surname_gray



def nameExtract(finalID):
    global DatosDNI, template_number
    if template_number == 2:
        name = finalID[114:132, 165:300] # DNI nuevo
    else:
        name = finalID[114:140, 165:305] # 220:320, 148:183
    name_gray = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/name.jpg", name_gray)
    cv2.resize(name_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(name_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el nombre")
                DatosDNI += "NOMBRE: \n"
            else:
                print(f"Nombre: {texto}")
                DatosDNI += f"NOMBRE: {texto}\n"
        except:
            print("No se ha podido leer el nombre (exception)")
    return name_gray


def numberExtract(finalID):
    global DatosDNI, template_number
    if template_number == 2:
        IDnumber = finalID[40:68, 180:325]
    else:
        IDnumber = finalID[238:265, 34:160]
    numero_gray = cv2.cvtColor(IDnumber, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/IDnumber.jpg", numero_gray)
    cv2.resize(numero_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(numero_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el número de DNI")
                DatosDNI += "NUMERO DE DNI: \n"
            else:
                print(f"Número de DNI: {texto}")
                DatosDNI += f"NUMERO DE DNI: {texto}\n"
        except:
            print("No se ha podido leer el número de DNI (exception)")

    return numero_gray

def signatureExtract(finalID):
    global template_number
    if template_number == 2:
        signature = finalID[200:250, 175:300]
    else:
        signature = finalID[200:255, 170:330]
    signature_gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/signature.jpg", signature_gray)
    cv2.resize(signature_gray, (0, 0), fx=6, fy=6)
    return signature_gray

def dueDateExtract(finalID):
    global DatosDNI, template_number
    if template_number == 2:
        dueDate = finalID[160:180, 245:330]
    else:
        dueDate = finalID[190:220, 240:330]
    dueDate_gray = cv2.cvtColor(dueDate, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/dueDate.jpg", dueDate_gray)
    cv2.resize(dueDate_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(dueDate_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer la fecha de caducidad")
                DatosDNI += "CADUCIDAD: \n"
            else:
                print(f"Fecha de caducidad: {texto}")
                DatosDNI += f"CADUCIDAD: {texto}\n"
        except:
            print("No se ha podido leer la fecha de caducidad (exception)")
    return dueDate_gray

def birthdayExtract(finalID):
    global DatosDNI, template_number
    if template_number == 2:
        birthday = finalID[135:155, 330:420]
    else:
        birthday = finalID[165:190, 170:260]
    birthday_gray = cv2.cvtColor(birthday, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/birthday.jpg", birthday_gray)
    cv2.resize(birthday_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(birthday_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer la fecha de nacimiento")
                DatosDNI += "NACIMIENTO: \n"
            else:
                print(f"Fecha de nacimiento: {texto}")
                DatosDNI += f"NACIMIENTO: {texto}\n"
        except:
            print("No se ha podido leer la fecha de nacimiento (exception)")

    return birthday_gray

def mrzExtract(finalID_backside):
    global DatosDNI
    mrz = finalID_backside[155:269, 1:424]
    mrz_gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/mrz.jpg", mrz_gray)
    cv2.resize(mrz_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(mrz_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el MRZ")
                DatosDNI += "MRZ: \n"
            else:
                print(f"MRZ: {texto}")
                DatosDNI += f"MRZ: {texto}\n"
        except:
            print("No se ha podido leer el MRZ (exception)")

    return mrz_gray


def main():
    '''
    temp1 = cv2.imread("DNITemplateFrontal_1.jpg")
    temp1 = cv2.resize(temp1, (425, 270))
    temp2 = cv2.imread("DNITemplateFrontal_2.jpg")
    temp2 = cv2.resize(temp2, (425, 270))

    cv2.imshow("Template 1", temp1)
    cv2.imshow("Template 2", temp2)

    cv2.waitKey()
    '''

    global rotation_matrix, control_frame, redo, template_number, webcam, DatosDNI, blursize
    capture = cv2.VideoCapture(webcam)

    DatosDNI = "---------------------------------------------------------------\n" \
               "|                         DATOS DNI                            |\n" \
               "---------------------------------------------------------------\n"

    while True:
        ret, frame = capture.read()
        ret, crop = capture.read()

        frame = detectionID(frame)
        areaScreen = frame.shape[0] * frame.shape[1]
        #blursize = (areaScreen) * (3 / 307200)
        #blursize = int(blursize)
        blursize = 3

        cv2.namedWindow("Normal Video")
        cv2.imshow("Normal Video", frame)

        if control_frame == True:
            print("Presione cualquier tecla para continuar...")
            cv2.waitKey()
            break
        else:
            if redo == True:
                cont_frame = 0
            if cv2.waitKey(30) >= 0:
                break

    xmax = vertex[0][0]
    xmin = vertex[0][0]
    ymax = vertex[0][1]
    ymin = vertex[0][1]

    for i in range(4):
        if xmax <= vertex[i][0]:
            xmax = vertex[i][0]
        if ymax <= vertex[i][1]:
            ymax = vertex[i][1]
        if xmin >= vertex[i][0]:
            xmin = vertex[i][0]
        if ymin >= vertex[i][1]:
            ymin = vertex[i][1]

    print(f"xmax: {xmax}, xmin: {xmin}, ymax: {ymax}, ymin: {ymin}")

    cutID = crop[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    output_directory = "Imagenes"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    cv2.imwrite("Imagenes/cutID.jpg", cutID)
    cv2.imshow("IDdetect", cutID)

    center = (cutID.shape[1] - 1) / 2.0, (cutID.shape[0] - 1) / 2.0

    # ----------------------------Case ID rotated to the right----------------------------------
    if heightRectangle < widhtRectangle and angle >= 60 and angle <= 90:
        print("DNI rotated to the right")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle-90, 1.0)

    # ----------------------------Case ID rotated to the left----------------------------------
    elif heightRectangle < widhtRectangle and angle >= 0 and angle <= 30:
        print("DNI rotated to the left")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    else:
        print("No se ha detectado el DNI correctamente, por favor reinicie el programa.")
        exit()

    # We rotate the image using warpAffine
    print(f"angle: {angle}")
    print(f"cutID.shape[0]: {cutID.shape[0]}, cutID.shape[1]: {cutID.shape[1]}")
    turnedImage = cv2.warpAffine(cutID, rotation_matrix, (cutID.shape[1], cutID.shape[0]))
    cv2.imshow("DNI rotado en uprigth position", turnedImage)
    cv2.imwrite("Imagenes/turnedImage.jpg", turnedImage)

    # We remove unwanted external edges from the rotated ID and resize to handle a known image size
    # If the ID is at a correct angle from the beginning (no garbage edges in the image) we only resize, no need to cut more image
    if (angle <= 2 and angle >= 0) or (angle >= 88 and angle <= 90):
        finalID = turnedImage
        finalID = cv2.resize(finalID, (425, 270)) # 365, 575
    # If these "garbage" edges exist we call IDrescale() and cut off the excess edges
    else:
        finalID = IDrescale(turnedImage)
    cv2.imshow("DNI", finalID)

    template_number = matchTemplate(finalID) # 1: DNI viejo, 2: DNI nuevo
    cv2.imwrite("Imagenes/finalID.jpg", finalID)
    # We get all the data from the front part of the ID in different images
    faceExtract(finalID)
    cv2.imshow("ID Number", numberExtract(finalID))
    cv2.imshow("Name", nameExtract(finalID))
    cv2.imshow("Surname", surnameExtract(finalID))
    cv2.imshow("signature", signatureExtract(finalID))
    cv2.imshow("Due date", dueDateExtract(finalID))
    cv2.imshow("Birthday date", birthdayExtract(finalID))
    cv2.waitKey(0)


    # REVERSO DEL DNI _________________________________________________________________
    control_frame = False
    redo = False
    messageID = False
    cont_frame = 0

    capture = cv2.VideoCapture(webcam)

    frame, frame_back = None, None
    while True:
        ret, frame_back = capture.read()
        ret, crop = capture.read()
        frame_back = detectionID(frame_back)

        cv2.namedWindow("Normal Video")
        cv2.imshow("Normal Video", frame_back)

        if control_frame == True:
            print("Presione cualquier tecla para continuar...")
            cv2.waitKey()
            break
        else:
            if redo == True:
                cont_frame = 0
            if cv2.waitKey(30) >= 0:
                break

    # We use the positions of the vertices of the ID (green rectangle) to segment it from the rest of the frame:
    xmax = vertex[0][0]
    xmin = vertex[0][0]
    ymax = vertex[0][1]
    ymin = vertex[0][1]

    for i in range(4):
        # UNCOMMENT TO SEE GREEN RECTANGLE
        # cv2.line(crop, vertex[i], vertex[(i + 1) % 4], (0, 255, 0), 3)

        if xmax <= vertex[i][0]:
            xmax = vertex[i][0]
        if ymax <= vertex[i][1]:
            ymax = vertex[i][1]
        if xmin >= vertex[i][0]:
            xmin = vertex[i][0]
        if ymin >= vertex[i][1]:
            ymin = vertex[i][1]

    print(f"xmax: {xmax}, xmin: {xmin}, ymax: {ymax}, ymin: {ymin}")

    # cv2.imshow("DNI capture", crop) # green square

    # We rotate in a new image the ID already segmented from the rest of the image:
    cutID = crop[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
    cv2.imwrite("Imagenes/cutIDBack.jpg", cutID)
    # cutID = crop.crop((left, top, right, bottom))
    cv2.imshow("IDdetect", cutID)

    center = (cutID.shape[1] - 1) / 2.0, (cutID.shape[0] - 1) / 2.0  # We find the center of the ID

    # ----------------------------Case ID rotated to the right----------------------------------
    if heightRectangle < widhtRectangle and angle >= 60 and angle <= 90:
        print("DNI rotated to the right")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)

    # ----------------------------Case ID rotated to the left----------------------------------
    elif heightRectangle < widhtRectangle and angle >= 0 and angle <= 30:
        print("DNI rotated to the left")
        # using getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    else:
        print("No se ha detectado el DNI correctamente, por favor reinicie el programa.")
        exit()

    # We rotate the image using warpAffine
    print(f"angle: {angle}")
    print(f"cutID.shape[0]: {cutID.shape[0]}, cutID.shape[1]: {cutID.shape[1]}")
    turnedImage = cv2.warpAffine(cutID, rotation_matrix, (cutID.shape[1], cutID.shape[0]))
    cv2.imshow("DNI rotado en uprigth position", turnedImage)
    cv2.imwrite("Imagenes/turnedImage_back.jpg", turnedImage)

    # We remove unwanted external edges from the rotated ID and resize to handle a known image size
    # If the ID is at a correct angle from the beginning (no garbage edges in the image) we only resize, no need to cut more image
    if (angle <= 2 and angle >= 0) or (angle >= 88 and angle <= 90):
        finalID = turnedImage
        finalID = cv2.resize(finalID, (425, 270))  # 365, 575
    # If these "garbage" edges exist we call IDrescale() and cut off the excess edges
    else:
        finalID = IDrescale(turnedImage)
    cv2.imshow("DNI", finalID)
    cv2.imwrite("Imagenes/finalID_back.jpg", finalID)
    # We get all the data from the back part of the ID in different images
    cv2.imshow("MRZ", mrzExtract(finalID))

    cv2.waitKey(0)
    # Guardamos los datos en un fichero
    with open("DatosDNI.txt", "w") as f:
        f.write(DatosDNI)

    # Cerrar programa
    print("Cerrando programa... Hasta pronto!")
    return 0


if __name__ == "__main__":
    main()