import argparse
import csv
import os
import platform
import sys
import pytesseract
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

global tesseractAvailable
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

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
        return facecrop
    except:
        print("No se ha podido encontrar la cara, revise la iluminación")


def surnameExtract(finalID):
    surname = finalID[80:110, 165:300] # 220:345, 90:150
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
            else:
                print(f"Apellido: {texto}")
        except:
            print("No se ha podido leer el apellido (exception)")

    return surname_gray



def nameExtract(finalID):
    name = finalID[114:132, 165:300] # 220:320, 148:183
    name_gray = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/name.jpg", name_gray)
    cv2.resize(name_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(name_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el nombre")
            else:
                print(f"Nombre: {texto}")
        except:
            print("No se ha podido leer el nombre (exception)")
    return name_gray


def numberExtract(finalID):
    IDnumber = finalID[40:68, 180:325]
    # IDnumber = finalID[238:261, 38:160]
    numero_gray = cv2.cvtColor(IDnumber, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/IDnumber.jpg", numero_gray)
    cv2.resize(numero_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(numero_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el número de DNI")
            else:
                print(f"Número de DNI: {texto}")
        except:
            print("No se ha podido leer el número de DNI (exception)")

    return numero_gray

def signatureExtract(finalID):
    signature = finalID[200:250, 175:300]
    signature_gray = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/signature.jpg", signature_gray)
    cv2.resize(signature_gray, (0, 0), fx=6, fy=6)
    return signature_gray

def dueDateExtract(finalID):
    dueDate = finalID[160:180, 245:330]
    dueDate_gray = cv2.cvtColor(dueDate, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/dueDate.jpg", dueDate_gray)
    cv2.resize(dueDate_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(dueDate_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer la fecha de caducidad")
            else:
                print(f"Fecha de caducidad: {texto}")
        except:
            print("No se ha podido leer la fecha de caducidad (exception)")
    return dueDate_gray

def birthdayExtract(finalID):
    birthday = finalID[135:155, 330:420]
    birthday_gray = cv2.cvtColor(birthday, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/birthday.jpg", birthday_gray)
    cv2.resize(birthday_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(birthday_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer la fecha de nacimiento")
            else:
                print(f"Fecha de nacimiento: {texto}")
        except:
            print("No se ha podido leer la fecha de nacimiento (exception)")

    return birthday_gray

def mrzExtract(finalID_backside):
    mrz = finalID_backside[155:269, 1:424]
    mrz_gray = cv2.cvtColor(mrz, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Imagenes/mrz.jpg", mrz_gray)
    cv2.resize(mrz_gray, (0, 0), fx=6, fy=6)

    if tesseractAvailable:
        try:
            texto = pytesseract.image_to_string(mrz_gray, config="--oem 3 --psm 6")
            if texto == "":
                print("No se ha podido leer el MRZ")
            else:
                print(f"MRZ: {texto}")
        except:
            print("No se ha podido leer el MRZ (exception)")

    return mrz_gray

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=20,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = 1
    source = str(source)
    front = True
    tesseractAvailable = True

    areaScreen = 0
    area = 0
    ratioaspecto = 0

    dni_count = 0

    # Directories
    dirImagenes = increment_path(ROOT / "Ejecucion" / "ejec", exist_ok=exist_ok)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    view_img = check_imshow(warn=True)
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            restart = True
            seen += 1
            w = 0
            h = 0

            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f"{i}: "

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            #print(f"Tamaño de la ventana: {im0.shape[0]}x{im0.shape[1]}")
            areaScreen = im0.shape[0] * im0.shape[1]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    if(names[int(c)] == "cell phone" or names[int(c)] == "book"):
                        restart = False

                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class

                        if restart:
                            draw_text(im0, "Muestre el DNI", font_scale=1, pos=(10, 10), text_color=(0, 0, 0),
                                      text_color_bg=(255, 255, 255))
                            dni_count = 0
                        else:
                            if names[c] == "cell phone" or names[c] == "book":
                                label = "DNI"
                                #print(f"DNI count: {dni_count}")
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            #annotator.box_label(xyxy, label, color=colors(c, True))
                                #areaDNI = xyxy[0] * xyxy[1]
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                                print(f"xywh: {xywh[0]}x{xywh[1]}x{xywh[2]}x{xywh[3]}   Area: {xywh[2] * xywh[3]}        Ratio: {xywh[2] / xywh[3]}")

                                area = xywh[2] * xywh[3]
                                ratioaspecto = xywh[2] / xywh[3]
                                #area = imc.shape[0] * imc.shape[1]
                                #ratioaspecto = imc.shape[0] / imc.shape[1]
                                #print(f"Area DNI: {area}, Ratio: {ratioaspecto}")

                                if xywh[2] < xywh[3]:
                                    draw_text(im0, "Asegurese de poner el DNI en horizontal", font_scale=1, pos=(10, 10),
                                              text_color=(0, 0, 0), text_color_bg=(255, 255, 255))
                                    #print("Asegurese de poner el DNI en horizontal")
                                    dni_count = 0
                                elif area < (areaScreen/10):
                                    draw_text(im0, "Acerque el DNI, por favor", font_scale=1, pos=(10, 10), text_color=(0,0,0),
                                              text_color_bg=(255, 255, 255))
                                    #print("Acerque el DNI, por favor")
                                    dni_count = 0
                                elif not (ratioaspecto > 1.45 and ratioaspecto < 1.63):
                                    draw_text(im0, "Enfoque bien el DNI, por favor", font_scale=1, pos=(10, 10),
                                              text_color=(0, 0, 0), text_color_bg=(255, 255, 255))
                                    #print("Enfoque bien el DNI, por favor")
                                    #dni_count = 0
                                    if dni_count > 0:
                                        dni_count -= 1
                                else:
                                    if (dni_count == 6 and front) or (dni_count == 4 and not front):
                                        draw_text(im0, f"Pulse R para repetir la foto", font_scale=1, pos=(10, 10),
                                                  text_color=(0, 0, 0), text_color_bg=(255, 255, 255))
                                        #print(f"Mostrando sus datos...")
                                    else:
                                        draw_text(im0, f"Mantenga...", font_scale=1, pos=(10, 10),
                                                  text_color=(0, 0, 0), text_color_bg=(255, 255, 255))
                                        #print(f"Mantenga ({6 - dni_count})...")
                                    dni_count += 1

                                repetirFront = False
                                repetirBack = False
                                if(dni_count == 8 and front):
                                        save_one_box(xyxy, imc, file= dirImagenes / f"DNIdelantero.jpg", BGR=True)
                                        dniDefinitivo = cv2.imread(dirImagenes / f"DNIdelantero.jpg")


                                        # Si se clica la leetra R se puede repetir la foto
                                        if cv2.waitKey(1) & 0xFF == ord("r"):
                                            repetirFront = True

                                        if not repetirFront:
                                            x_inicio = (dniDefinitivo.shape[1] - xywh[2] + 9) // 2
                                            y_inicio = (dniDefinitivo.shape[0] - xywh[3] + 9) // 2
                                            x_fin = x_inicio + xywh[2] - 9
                                            y_fin = y_inicio + xywh[3] - 9

                                            dniDefinitivo = dniDefinitivo[int(y_inicio):int(y_fin), int(x_inicio):int(x_fin)]
                                            dniDefinitivo = cv2.resize(dniDefinitivo, (425, 270))
                                            if not os.path.exists("Imagenes"):
                                                os.makedirs("Imagenes")

                                            cv2.imwrite(f"Imagenes/ParteDelantera.png", dniDefinitivo)
                                            cv2.imshow("Parte delantera", dniDefinitivo)
                                            cv2.waitKey()
                                            cv2.imshow("Face ID", faceExtract(dniDefinitivo))
                                            cv2.imshow("ID Number", numberExtract(dniDefinitivo))
                                            cv2.imshow("Name", nameExtract(dniDefinitivo))
                                            cv2.imshow("Surname", surnameExtract(dniDefinitivo))
                                            cv2.imshow("signature", signatureExtract(dniDefinitivo))
                                            cv2.imshow("Due date", dueDateExtract(dniDefinitivo))
                                            cv2.imshow("Birthday date", birthdayExtract(dniDefinitivo))
                                            cv2.waitKey()
                                            front = False

                                if(dni_count == 6 and not front):
                                        save_one_box(xyxy, imc, file=dirImagenes / f"DNItrasero.jpg", BGR=True)
                                        dniDefinitivo = cv2.imread(dirImagenes / f"DNItrasero.jpg")

                                        # Si se clica la leetra R se puede repetir la foto
                                        if cv2.waitKey(1) & 0xFF == ord("r"):
                                            repetirBack = True

                                        if not repetirBack:
                                            x_inicio = (dniDefinitivo.shape[1] - xywh[2] + 9) // 2
                                            y_inicio = (dniDefinitivo.shape[0] - xywh[3] + 9) // 2
                                            x_fin = x_inicio + xywh[2] - 9
                                            y_fin = y_inicio + xywh[3] - 9

                                            dniDefinitivo = dniDefinitivo[int(y_inicio):int(y_fin),
                                                            int(x_inicio):int(x_fin)]
                                            dniDefinitivo = cv2.resize(dniDefinitivo, (425, 270))
                                            cv2.imwrite(f"Imagenes/ParteTrasera.png", dniDefinitivo)
                                            cv2.imshow("Parte trasera", dniDefinitivo)
                                            cv2.waitKey()
                                            cv2.imshow("MRZ", mrzExtract(dniDefinitivo))
                                            cv2.waitKey()
                                            return 0

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                cv2.imshow(str(p), im0)

                if dni_count == 30:
                    print("Presione cualquier tecla para continuar...")
                    cv2.waitKey()  # 1 millisecond
                cv2.waitKey(1)  # 1 millisecond
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)