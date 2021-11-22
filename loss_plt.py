# -*- coding utf-8 -*-
"""
Create on 2021/1/25 14:01
@author: zsw
"""
from matplotlib import pyplot as plt
import numpy as np
import re


def main(txts):
    result_list = re.findall(r"[=](.*?)[(]", txts)
    loss_list = [float(loss) for loss in result_list]

    plt.title('LOSS')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.plot(range(0,len(loss_list)),loss_list)
    plt.show()




if __name__ == '__main__':
    with open('E:\Desktop\PAPER\\bert-master\cnews\\loss.txt','r') as f:
        txts = f.read()
    main(txts[14018:])