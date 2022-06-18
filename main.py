# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# git提交代码跳过ssl验证
##git config --global http.sslVerify false
import os

import cv2
import numpy as np
import cv2 as cv
import shutil
from BatchRename import BatchRename
import matplotlib.pyplot as plt
import matplotlib


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


#Sobel算子边缘检测
def SobelEdge(img):
    # 计算Sobel卷积结果
    x = cv2.Sobel(img, cv.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv.CV_16S, 0, 1)
    # 转换数据 并 合成
    Scale_absX = cv2.convertScaleAbs(x)  # 格式转换函数
    Scale_absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(Scale_absX, 0.7, Scale_absY, 0.7, 0)  # 图像混合

#Canny算子边缘检测
def CannyEdge(img,parm1,parm2):
    return cv2.Canny(img,parm1,parm2)

#Roberts算子
def RobertsEdge(img):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv.filter2D(img, cv.CV_16S, kernelx)
    y = cv.filter2D(img, cv.CV_16S, kernely)

#Prewitt算子

#Scharr算子

#Krisch算子


#命名窗口大小
def NameWindow(windosName,x,y):
    cv2.namedWindow(windosName,0)
    return cv2.resizeWindow(windosName,x,y)

#二值化图片
def ThreshHoldImg(img,thresh,maxval=255):
    ret, mask_all = cv2.threshold(src=img,  # 要二值化的图片
                                  thresh=thresh,  # 全局阈值
                                  maxval=maxval,  # 大于全局阈值后设定的值
                                  type=cv2.THRESH_BINARY)  # 设定的二值化类型，THRESH_BINARY：表示小于阈值置0，大于阈值置填充色
    return mask_all
    print("全局阈值的shape: ", mask_all.shape)

#膨胀
def dilate_demo(img):
    #ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))#定义结构元素的形状和大小
    dst = cv2.dilate(img, kernel,iterations=1)#膨胀操作
    return dst

#边缘算法后批量写文件
def BatchWrite(srcImgDir,dstImgDir,edgeAlgorithm):
    filelist = os.listdir(srcImgDir)
    print("filelist",filelist)
    total_num = len(filelist)
    i = 1
    for item in filelist:
        if(i>total_num):
            break
        else:
            src = os.path.join(os.path.abspath(srcImgDir), '' + item)  # 当前文件中图片的地址
            print("srcimg:" + src)
            # 读灰度图
            gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            # 高斯滤波
            gray = cv2.GaussianBlur(gray, (5, 5), 5)
            # 中值滤波
            gray = cv2.medianBlur(gray, 7)

            if(edgeAlgorithm == ThreshHoldImg):   #全局阈值法
                result = ThreshHoldImg(gray, 110)
            elif(edgeAlgorithm ==  CannyEdge):    #canny算子
                result = CannyEdge(gray,50,100)
            elif(edgeAlgorithm == SobelEdge):
                result = SobelEdge(gray)          #Sobel算法
                result = dilate_demo(result)
            dst = dstImgDir + str(item)
            print("dst:"+ dst)

            cv2.imwrite(dst,result)
            i = i + 1

if __name__ == '__main__':
    print_hi('PyCharm')

    #BatchWrite('D:/PyCharm/Projects/pythonProject1/Pics1/','D:/PyCharm/Projects/pythonProject1/result2/',SobelEdge)

    #读灰度图
    gray = cv2.imread('Pics1/1.jpg',cv2.IMREAD_GRAYSCALE)
    #高斯滤波
    gray = cv2.GaussianBlur(gray,(9, 9), 9)
    #中值滤波
    gray = cv2.medianBlur(gray,7)
    #sobel算子
    sobel = SobelEdge(gray)
    #膨胀
    #result = dilate_demo(sobel)

    #全局阈值法
    result = ThreshHoldImg(sobel,35)

    #canny算子
    #canny = CannyEdge(gray,50,100)
    #print("写文件"+str(i)+".jpg啦")
    #dst = os.path.join(os.path.abspath("D:\PyCharm\Projects\pythonProject1\result"), '' + str(i) + '.jpg')

    NameWindow("result",2448,2048)
    cv2.imshow("result",result)

    cv2.waitKey()
    cv2.destroyAllWindows()

    print("i am fine hhhhh!")
    # demo = BatchRename()
    # demo.rename()

