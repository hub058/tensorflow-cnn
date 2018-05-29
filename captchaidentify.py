'''
Created on 2018年5月26日

@author: Administrator
'''
import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

'''
打印英文字母
print(help(print))
for i in range(26):
    print("'"+chr(ord('A')+i)+"',",end="")
'''
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number+alphabet+ALPHABET,char_size=4):
    captcha_text =[]
    for _ in range(char_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    # 把list转成字符串
    captcha_text = ''.join(captcha_text)
    
    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text+".png")
    captcha_image = Image.open(captcha,'r')
    captcha_image = np.array(captcha_image)
    return captcha_text,captcha_image
    

if __name__ == '__main__':
#     text, image = gen_captcha_text_and_image()
#     # 画出验证码
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     ax.text(0.1,0.9,text,ha='center',va='center',transform=ax.transAxes)
#     plt.imshow(image)
#     plt.show()
    image =ImageCaptcha().generate("zhaoendong",'jpeg')
    image = Image.open(image,'r')
    plt.imshow(image)
    plt.show()





