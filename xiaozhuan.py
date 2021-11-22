# -*- coding utf-8 -*-
"""
Create on 2020/12/16 16:12
@author: zsw
"""

from PIL import Image, ImageFont, ImageDraw
from docx import Document

def main():
    texts = []
    with open('chinese_L-12_H-768_A-12\\vocab.txt',
              encoding='utf-8') as f:
        datas = f.readlines()
    for data in datas:
        text = data.rstrip('\n')
        texts.append(text)
    print(texts)
    print(len(texts))
    import os
    for roots,dirs,files in os.walk('小篆\\'):
        file_list = files
        break
    sign_dict = {}
    j = 0
    for txt in texts:
        sign_dict[str(j)] = txt
        j += 1
        if txt+'.png' in file_list:
            try:
                int(txt)
            except ValueError:
                os.rename("小篆\\"+txt+'.png',
                      "小篆\\" + str(j) + '.png')
                continue
        # im = Image.new("RGB", (200, 200), (255, 255, 255))
        im = Image.new("L", (100, 100), 255)
        im2 = im.copy()
        dr = ImageDraw.Draw(im)
        size = 100
        location = (0, 0)
        if len(txt) != 1:
            size = 20
            location = (0, 40)
        font = ImageFont.truetype("simhei.ttf", size)
        dr.text(location, txt, font=font, fill="#000000")
        # im.show()
        if im != im2:#去除无法识别的字体
            im.save("小篆\\"+str(j)+".png")
        print(j, txt)

    with open("小篆\\sign_dict.txt",'w',encoding='utf-8') as f:
        for key,value in sign_dict.items():
            txt = key + '\t' + value + '\n'
            f.write(txt)


if __name__ == '__main__':
    main()
    #建立空白字符
    im = Image.new("L", (100, 100), 255)
    dr = ImageDraw.Draw(im)
    size = 100
    location = (0, 0)
    font = ImageFont.truetype("simhei.ttf", size)
    dr.text(location, '', font=font, fill="#000000")
    im.save("小篆\\0.png")
