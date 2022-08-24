from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2


@dataclass
class Service:

    def convert_to_gs(path) -> Image:
        gray_image = Image.open(path)
        gray_image = gray_image.convert('L')
        print(gray_image.mode)
        return gray_image

    def color_at(img_arr, row, column) -> tuple:
        if img_arr.flags.writeable:
            return img_arr[row, column]

    def reduce_to(path, ch) -> Image:
        img = Image.open(path)
        img_arr = np.array(img)
        if ch == 'R' or ch == 'r':
            red_img_arr = img_arr.copy()
            red_img_arr[:, :, (1, 2)] = 0
            red_img_arr = Image.fromarray(red_img_arr)
            return red_img_arr
        elif ch == 'G' or ch == 'g':
            green_img_arr = img_arr.copy()
            green_img_arr[:, :, (0, 2)] = 0
            green_img_arr = Image.fromarray(green_img_arr)
            return green_img_arr
        elif ch == 'B' or ch == 'b':
            blue_img_arr = img_arr.copy()
            blue_img_arr[:, :, (0, 1)] = 0
            blue_img_arr = Image.fromarray(blue_img_arr)
            return blue_img_arr

    def make_college(imglst):
        rbglst = []
        for i in range(len(imglst)):
            for x in range(2):
                if i == len(imglst):
                    break
                red_img_arr = imglst[i].copy()
                red_img_arr = np.array(red_img_arr)
                red_img_arr[:, :, (1, 2)] = 0
                red_img_arr = Image.fromarray(red_img_arr)
                rbglst.append(red_img_arr)
                i += 1
            for x in range(2):
                if i == len(imglst):
                    break
                green_img_arr = imglst[i].copy()
                green_img_arr = np.array(green_img_arr)
                green_img_arr[:, :, (0, 2)] = 0
                green_img_arr = Image.fromarray(green_img_arr)
                rbglst.append(green_img_arr)
                i += 1
            for x in range(2):
                if i == len(imglst):
                    break
                blue_img_arr = imglst.copy()
                blue_img_arr = np.array(blue_img_arr)
                blue_img_arr[:, :, (0, 1)] = 0
                blue_img_arr = Image.fromarray(blue_img_arr)
                rbglst.append(blue_img_arr)
                i += 1
        img_gallery = np.concatenate(rbglst, axis=0)
        return img_gallery

    def shapes_dict(imglist):
        img_dict = {}
        for i, img in enumerate(imglist):
            img_arr = np.array(img.copy())
            i = i + 1
            img_dict.update({i: img_arr.shape})
        return img_dict

    def show_img(image_name):
        while True:
            cv2.imshow('image', image_name)
            key_pressed = cv2.waitKey(0)
            if key_pressed:
                break
        cv2.destroyAllWindows()

    def draw_green_rectangle(img, cascade):
        for (row, column, width, height) in cascade:
            cv2.rectangle(img,
                          (row, column),
                          (row + width, column + height),
                          (0, 255, 0),
                          2)

    def detect_obj(path, body_part) -> Image:
        img = cv2.imread(path)
        img_copy = img.copy()
        gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        if body_part.lower() == "face":
            face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_classifier.detectMultiScale(gray_img)
            Service.draw_green_rectangle(img_copy, faces)
        elif body_part.lower() == "eyes":
            eyes_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            eyes = eyes_classifier.detectMultiScale(gray_img)
            Service.draw_green_rectangle(img_copy, eyes)
        return img_copy

    def detect_obj_adv(path, face: bool, eyes: bool):
        img = cv2.imread(path)
        img_copy = img.copy()
        gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        if face:
            face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_classifier.detectMultiScale(gray_img)
            Service.draw_green_rectangle(img_copy, faces)
        if eyes:
            eyes_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            eyes = eyes_classifier.detectMultiScale(gray_img)
            Service.draw_green_rectangle(img_copy, eyes)
        return img_copy

    def detect_face_in_vid(path):
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        vid = cv2.VideoCapture(path)
        while True:
            ret, frame = vid.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            Service.draw_green_rectangle(frame, faces)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(0):
                break
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cat1 = Service.convert_to_gs('Cat.jpg')
    print(cat1)

    cat_arr = Image.open('Cat.jpg')
    cat_arr = np.array(cat_arr)
    print(Service.color_at(cat_arr, 200, 300))

    lst1 = [Image.open('Cat.jpg'), Image.open('img.jpg'), Image.open('Dog.jpg')]
    print(Service.shapes_dict(lst1))

    lst2 = [Image.open('img.jpg'), Image.open('img.jpg'), Image.open('img.jpg'), Image.open('img.jpg')]
    print(Service.make_college(lst2))

    # green_dog_img = Service.reduce_to('Dog.jpg', 'g')
    # green_dog_img.show()

    # Service.show_img(Service.detect_obj('img.jpg', "eyEs"))

    # Service.show_img(Service.detect_obj_adv('img.jpg', True, True))

    # Service.detect_face_in_vid('vid.mp4')
