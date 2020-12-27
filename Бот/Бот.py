import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
import tensorflow as tf
import numpy as np
import time
import datetime
import idx2numpy
from ctypes import windll, Structure, c_long, byref
from PIL import Image, ImageGrab
import cv2
from typing import *
import sys
import pyautogui
import msvcrt
import pickle
##
class POIN(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]
t=datetime.datetime.now()
stroka="Subscription expires on 12.09.200"
def cnn_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file, 
target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))

   result = model.predict_classes([img_arr])
   return result[0]
def queryMousePosition():
    pt = POIN()
    windll.user32.GetCursorPos(byref(pt))
    return pt                     
def WorkOnImage():
    image = 'screen2.png'
    preprocess = "thresh"

    # загрузить образ и преобразовать его в оттенки серого
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # проверьте, следует ли применять пороговое значение для предварительной обработки изображения

    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# если нужно медианное размытие, чтобы удалить шум
    elif preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

# сохраним временную картинку в оттенках серого, чтобы можно было применить к ней OCR

    filename = "screen2.png"
    cv2.imwrite(filename, gray)
    return filename
def TestImage():
  pos=[]
  for i in range(2):
    print("Image ",i)
    s='o'
    while(s!='y'):
        print("Position 1. Ready?(y)")
        if(input()=="y"):
            pos2 = queryMousePosition()
        print("Position 2. Ready?(y)")
        if(input()=="y"):
            pos1 = queryMousePosition()
        img3 = ImageGrab.grab( (pos2.x, pos2.y, pos1.x, pos1.y) )
        img3.show()
        print("ready?(y)")
        s=input()
    pos=pos+[pos2,pos1]        
  return pos
def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    #print('The original image size is {wide} wide x {height} ' 'high'.format(wide=width, height=height))
 
    resized_image = original_image.resize(size)
    width, height = resized_image.size
    #print('The resized image size is {wide} wide x {height} ' 'high'.format(wide=width, height=height))
    #resized_image.show()
    resized_image.save(output_image_path, "PNG")

def letters_extract(image_file: str, out_size=28) -> List[Any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=True)

    return letters
def img_to_str(model: Any, image_file):
    letters = letters_extract(image_file)
    s_out = 0
    razryd=1
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        cv2.imwrite("1.png", letters[i][2])
        s_out+=(cnn_digits_predict(model, "1.png")*razryd)
        razryd*=10
       # if (dn > letters[i][1]/4):
          #  s_out += ' '
    return s_out
def Digit(model: Any, pos, pos1, check,check_f):
      img2 = ImageGrab.grab( (pos[0].x, pos[0].y, pos[1].x, pos[1].y) )
      #img2 = ImageGrab.grab( (1287, 560,  1317, 581) )
      img2.save("screen2.png", "PNG")
      resize_image(input_image_path='screen2.png', output_image_path='screen2.png', size=(600, 600))
      #scale_image(input_image_path='screen2.png',output_image_path='screen3.png',width=1000, height=1000)
      filename = WorkOnImage()                                                                               #       4
      first=img_to_str(model,"screen2.png")
      img2 = ImageGrab.grab( (pos[2].x, pos[2].y, pos[3].x, pos[3].y) )
      #img2 = ImageGrab.grab( (1331, 559,  1353, 584) )
      img2.save("screen2.png", "PNG")
      resize_image(input_image_path='screen2.png', output_image_path='screen2.png', size=(600, 600))
      #scale_image(input_image_path='screen2.png',output_image_path='screen3.png',width=1000, height=1000)
      filename = WorkOnImage()                                                                               #       7
      second=img_to_str(model,"screen2.png")
      if(first<check_f):
          img2 = ImageGrab.grab( (pos1[0].x, pos1[0].y, pos1[1].x, pos1[1].y) )
          img2.save("screen2.png", "PNG")
          resize_image(input_image_path='screen2.png', output_image_path='screen2.png', size=(600, 600))
          filename = WorkOnImage()                                                                               #       4
          first1=img_to_str(model,"screen2.png")
          if(first1>=check_f):
            first=first1
            img2 = ImageGrab.grab( (pos1[2].x, pos1[2].y, pos1[3].x, pos1[3].y) )
            img2.save("screen2.png", "PNG")
            resize_image(input_image_path='screen2.png', output_image_path='screen2.png', size=(600, 600))
            filename = WorkOnImage()
            second=img_to_str(model,"screen2.png")
      if((first==0)and(second<=check)):
          first=second
          second=0
      price=first*100+second
      return price
def StartBot(t):
    with open("libary.dll","rb") as libary:
        check=pickle.load(libary)
    with open("key.dat","rb") as key:
        k=pickle.load(key)
        d=pickle.load(key)
        m=pickle.load(key)
        d_=pickle.load(key)
        m_=pickle.load(key)
    if(check==k):
        if(t.month>=m and t.month<(m_+1)):
            if((t.month!=m_)or(t.month==m_ and t.day<(d_+1))):
                print(stroka)
                return True
            else:
                print("Subscription has expired")
                return False
        else:
            print("Subscription has expired")
            return False
    else:
        print("Invalid key")
        return False
def Restart(k,pt):
    while msvcrt.kbhit():
        msvcrt.getch()
    if(input("Command?(y)")=="y"):
        time.sleep(k)
        pyautogui.press("g")
        time.sleep(0.2)
        pyautogui.press("enter")
        print("Restart")
        if(input("Ready?(g)")=="g"):
            time.sleep(1)
            print("Restart Go")
            pyautogui.moveTo(pt.x, pt.y, duration=1, tween=pyautogui.easeInOutQuad)
            pyautogui.click()
            pyautogui.write('y', interval=0.1)
            pyautogui.press("enter")
            time.sleep(0.3)
            pyautogui.moveTo(500, 500, duration=1, tween=pyautogui.easeInOutQuad)
            time.sleep(3)
    if(input("Ready?(y)")=="y"):
        print("GO")   
def Bought(k):
    while msvcrt.kbhit():
        msvcrt.getch()
    if(input("Command?(y)")=="y"):
        time.sleep(k)
        pyautogui.press("b")
        time.sleep(0.2)
        pyautogui.press("enter")
        print("Buy")
    while(input("Ready?(b)")!="b"):
        time.sleep(1)
        print("GO")
def Sell(k,t_sell,start_bot):
    if(time.time()-t_sell<60 and start_bot==False):
        time.sleep(60)
    while msvcrt.kbhit():
        msvcrt.getch()
    if(input("Command?(y)")=="y"):
        time.sleep(k)
        pyautogui.press("s")
        time.sleep(0.2)
        pyautogui.press("enter")
        print("Sell")
    while(input("Ready?(s)")!="s"):
        time.sleep(1)
        print("GO")
def Stop(k):
    while msvcrt.kbhit():
        msvcrt.getch()
    if(input("Command?(y)")=="y"):
        time.sleep(k)
        pyautogui.press("g")
        time.sleep(0.2)
        pyautogui.press("enter")
        print("Stop")
k=StartBot(t)
if(k==False):
   exit(0)
def Start():
    pt=POIN()
    if(input("Ready?(y)")=="y"):
        windll.user32.GetCursorPos(byref(pt))
        time.sleep(2)
        pyautogui.moveTo(pt.x, pt.y, duration=1, tween=pyautogui.easeInOutQuad)
        pyautogui.click()
        print(pt.x,pt.y)
        pyautogui.write('y', interval=0.1)
        pyautogui.press("enter")
        time.sleep(0.3)
        pyautogui.moveTo(500, 500, duration=1, tween=pyautogui.easeInOutQuad)
        time.sleep(3)
    if(input("Ready?(y)")=="y"):
        print("GO")
    return pt
#print("1 Point: ", pos[0].x,pos[0].y)
#print("2 Point: ", pos[1].x,pos[1].y)
#print("3 Point: ", pos[2].x,pos[2].y)
#print("4 Point: ", pos[3].x,pos[3].y)
#Координаты первого лота цифр
with open("data.pickle","rb") as data:
    first_1=pickle.load(data)
    first_2=pickle.load(data)
    second_1=pickle.load(data)
    second_2=pickle.load(data)
if __name__ == '__main__':
   #try:
   price_buy=int(input("Price Buy:"))
   price_sell=int(input("Sell Buy:"))
   check_pr=int(input("Check bronze or silver:"))
   check_b=price_buy//100
   buy_count=0
   i=int(input("Time(<=48 hours):"))
   time_full=48*3600
   pt_re=Start()
   k=0.3
   model = tf.keras.models.load_model('cnn_digits_28x28(3).h5')   
   t_start=time.time()
   t_restart=t_start
   t_sell=time.time()
   start_bot=True
   while(time.time()-t_start<i and time.time()-t_start<time_full):
        price_1=Digit(model,first_1,first_2,check_pr,check_b)
        price_2=Digit(model,second_1,second_2,check_pr,check_b)
        if(price_1<=price_buy and price_2<=price_buy):
            Bought(k)
            buy_count+=1
        if(price_1>price_sell and price_2>price_sell):
            Sell(k,t_sell,start_bot)
            t_sell=time.time()
            start_bot=False
        if(time.time()-t_restart>3600):
            Restart(k,pt_re)
            t_restart=time.time()
        print("Price 1: ", price_1)
        print("Price 2: ", price_2)
        print("Bought: ", buy_count)
        print("Time: ",time.time()-t_start)
   Stop(k)
   input()
   time.sleep(3)