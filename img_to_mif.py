'''
Este programa converte uma imagem
em RGB para um arquivo .mif
'''
import cv2 as cv
import os 
import numpy as np

# definicoes arquitetura memoria
WIDTH=8
DEPTH=1024

def cabecalho ():
    str_ret = ""
    str_ret += "-- begin_signature\n"
    str_ret += "-- ROM\n"
    str_ret += "-- end_signature\n"
    str_ret += "WIDTH="+str(WIDTH)+";\n"
    str_ret += "DEPTH="+str(DEPTH)+";\n"
    str_ret += "ADDRESS_RADIX=UNS;\n"
    str_ret += "DATA_RADIX=BIN;\n\n"
    str_ret += "CONTENT BEGIN\n"
    print(str_ret)
    return str_ret
    

def rodape():
    str_ret = ""
    str_ret += "\nEND;\n"
    print(str_ret)
    return str_ret


im = cv.imread("Y_74978.png")
im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
imw = im_rgb.shape[1]       # largura
imh = im_rgb.shape[0]       # altura
num_pix_chanel = imh * imw


index = 0
with open("input_chanel_R.mif", "w") as mif:    
    mif.writelines(cabecalho()) 

    for i in range(DEPTH, num_pix_chanel, -1):
        mif.writelines(str(DEPTH-index-1) + ": 00000000; \n")
        index += 1

    for linha in range(imh-1, -1, -1):
        for coluna in range(imw-1, -1, -1):
            pixel = im_rgb[linha, coluna, 0]
            bin_value = bin(pixel).replace("0b", "")
            bin_value = bin_value.zfill(WIDTH)           
            mif.writelines(
                    str(DEPTH-index-1) +
                    ": " + bin_value +
                    "; \n")
            print(DEPTH-index-1, ": ", bin_value)
            index += 1
    
    mif.writelines(rodape())

index = 0
with open("input_chanel_G.mif", "w") as mif:   
    mif.writelines(cabecalho())

    for i in range(DEPTH, num_pix_chanel, -1):
        mif.writelines(str(DEPTH - index - 1) + ": 00000000; \n")
        index += 1

    for linha in range(imh - 1, -1, -1):
        for coluna in range(imw - 1, -1, -1):
            pixel = im_rgb[linha, coluna, 1]
            bin_value = bin(pixel).replace("0b", "")
            bin_value = bin_value.zfill(WIDTH)           
            mif.writelines(
                    str(DEPTH-index-1) +
                    ": " + bin_value +
                    "; \n")
            print(DEPTH-index-1, ": ", bin_value)
            index += 1
    
    mif.writelines(rodape())

index = 0
with open("input_chanel_B.mif", "w") as mif:    
    mif.writelines(cabecalho())

    for i in range(DEPTH, num_pix_chanel, -1):
        mif.writelines(str(DEPTH - index - 1) + ": 00000000; \n")
        index += 1

    for linha in range(imh - 1, -1, -1):
        for coluna in range(imw - 1, -1, -1):
            pixel = im_rgb[linha, coluna, 2]
            bin_value = bin(pixel).replace("0b", "")
            bin_value = bin_value.zfill(WIDTH)           
            mif.writelines(
                    str(DEPTH-index-1) +
                    ": " + bin_value +
                    "; \n")
            print(DEPTH-index-1, ": ", bin_value)
            index += 1
    
    mif.writelines(rodape())


cv.imshow("Red", im_rgb[..., 0])
cv.imshow("Green", im_rgb[..., 1])
cv.imshow("Blue", im_rgb[..., 2])

cv.waitKey(0)