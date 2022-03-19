import cv2
from math import ceil
import os

cap = cv2.VideoCapture(0) # 0 -> first webcam

st = "".join(list(reversed('Ñ@#W$9876543210?!abc;:+=-,._  ')))

while(True):
    os.system('cls' if os.name == 'nt' else 'clear')
    ret, frame = cap.read() # ret -> return: True or False
    
    # img'ı resize et
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(gray, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gray image', img)
  
    # for döngüsü ile parlaklık bul
        # parlaklığa göre karakter yazdır
    for i in img:
        for j in i:
            print(st[min(ceil(j/200 * len(st)), len(st)-1)], end='')
        print()

    
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
