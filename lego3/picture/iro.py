import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4, A5, portrait, landscape
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display_pdf
from scipy.stats import rankdata
import scipy
import os
def _convolve2d(image, kernel):
    shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1) + kernel.shape
    strides = image.strides * 2
    strided_image = np.lib.stride_tricks.as_strided(image, shape, strides)
    return np.einsum('kl,ijkl->ij', kernel, strided_image)
def _convolve2d_multichannel(image, kernel):
    convolved_image = np.empty((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, image.shape[2]))
    for i in range(image.shape[2]):
        convolved_image[:,:,i] = _convolve2d(image[:,:,i], kernel)
    return convolved_image
def _pad_singlechannel_image(image, kernel_shape, boundary):
    return np.pad(image, ((int(kernel_shape[0] / 2),), (int(kernel_shape[1] / 2),)), boundary)
def convolve2d(image, kernel, boundary='edge'):
    if image.ndim == 2:
        pad_image = _pad_singlechannel_image(image, kernel.shape, boundary) if boundary is not None else image
        return _convolve2d(pad_image, kernel)
def create_averaging_kernel(size = (3, 3)):
    return np.full(size, 1 / (size[0] * size[1]))
def median1d(arr, k=3):
    w = len(arr)
    idx = np.fromfunction(lambda i, j: i + j, (k, w), dtype=np.int) - k // 2
    idx[idx < 0] = 0
    idx[idx > w - 1] = w - 1
    result=np.median(arr[idx], axis=0)
    result = result.astype(int)
    return result
def median2d(arr):
    result=scipy.signal.medfilt2d(arr, kernel_size=3)
    #result = result.astype(int)
    return result
"""
深度マップを、指定した高さに量子化する関数(depth:深度マップ、ブロックの最大の高さn)
"""
def posterization(depth, n):
    #print(depth)
    depth = 255-depth
    th_bin, depth_bin = cv2.threshold(depth,0,255,cv2.THRESH_OTSU)
    ran2 = th_bin
    out = np.ones_like(depth)
    out = np.array(out.flatten())
    depth_onedim=depth.flatten()
    rank=np.array(rankdata(depth_onedim))#手前順ランキング
    back=depth_onedim[ran2<depth_onedim].shape[0]#背景の画素数
    ran1 = (depth.shape[0]*depth.shape[1]-back)/(n-1)
    for i in range(n-1):
        out=np.where((i*ran1<=rank)&(rank<(i+1)*ran1),n-i,out)
    out=np.reshape(out,(depth.shape[0],depth.shape[1]))
    averaging_kernel_3x3 = create_averaging_kernel(size=(3, 3))
    out = convolve2d(out, averaging_kernel_3x3)
    out=out.astype(int)#ここコメントアウトしたら滑らかになる（少数を許す）
    return out

"""使用するブロックの色を表示する関数(color_dic={"color_name":[R,G,B]})"""
def show_col_list(color_dic):
    color_rgb = np.array([[c_l] for c_l in color_dic.values()]).astype(np.uint8)
    color_rgb = color_rgb.transpose(1, 0, 2)
    plt.figure(figsize=(20,10))
    plt.imshow(color_rgb[...,::-1])
    plt.show()
"""LEGOブロックの色のみに変換する関数(モザイクがかけられたRGB画像、使用するブロックの色の辞書)"""
def change_coler(img_rgb, color_dic):
    Height, Width = img_rgb.shape[:2]
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    color_dic_rgb = np.array([[c_l] for c_l in color_dic.values()]).astype(np.uint8)
    color_dic_rgb = color_dic_rgb.transpose(1, 0, 2)
    color_dic_lab = cv2.cvtColor(color_dic_rgb, cv2.COLOR_BGR2LAB).astype(np.int16)
    
    
    img_L, img_A, img_B = img_lab[...,0], img_lab[...,1], img_lab[...,2]
    list_L, list_A, list_B = color_dic_lab[...,0], color_dic_lab[...,1], color_dic_lab[...,2]
    list_L, list_A, list_B = np.ravel(list_L), np.ravel(list_A), np.ravel(list_B)
    
    """
    RGBで比較
    img_L, img_A, img_B = img_rgb[...,0], img_rgb[...,1], img_rgb[...,2]
    list_L, list_A, list_B = col_list_rgb[...,0], col_list_rgb[...,1], col_list_rgb[...,2]
    list_L, list_A, list_B = np.ravel(list_L), np.ravel(list_A), np.ravel(list_B)
    """
    #各ブロックの色との差を格納
    diff = np.zeros((Height, Width, len(color_dic)))
    #一番差が小さいブロックの色のインデックスを格納
    min_col_num = np.zeros((Height, Width))

    for h in range(Height):
        for w in range(Width):
            diff[h,w] = ((float(img_L[h,w]) - list_L)**2 + (float(img_A[h,w]) - list_A)**2 + (float(img_B[h,w]) - list_B)**2)**0.5
            min_col_num[h,w] = np.argmin(diff[h,w])
    
    col_keys = list(color_dic.keys())
    Out = np.zeros_like(img_rgb)
    for h in range(Height):
        for w in range(Width):
            Out[h,w] = color_dic[col_keys[int(min_col_num[h,w])]]
    
    return Out,min_col_num
"""RGBとdepth情報をまとめたnumpy配列を作製する関数"""
def create_rgb_depth_map(rgb, depth):
    depth = depth[:,:,np.newaxis]
    print(rgb.shape)
    print(depth.shape)
    Out = np.concatenate([rgb,depth],axis=2)
    return Out

"""depth情報と色情報をそれぞれCSVファイルに保存する"""
def create_csv_file(rgb, depth):
    Height, Width = rgb.shape[:2]
    rgb_map = [[] for _ in range(Height)]
    for h in range(Height):
        for w in range(Width):
            rgb_merge = hex(rgb[h,w,2])[2:] + hex(rgb[h,w,1])[2:] + hex(rgb[h,w,0])[2:]
            rgb_map[h].append("0x" + rgb_merge)
    
    df_rgb = pd.DataFrame(rgb_map)
    df_rgb.to_csv("rgb_map.csv", header=False, index=False)
    df_depth = pd.DataFrame(depth)
    df_depth.to_csv("depth_map.csv", header=False, index=False)
"""
説明書PDFを作成する関数
color_map: LEGOで作成する最終的なカラー画像
depth_map: LEGOで作成する最終的な深度画像
min_color_num: 関数'change_coler'で返される
COLOR_DIC: 使用する色の辞書型配列
"""
def creat_instruction(color_map, depth_map, min_color_num, COLOR_DIC,image_id):
    table_len = 15
    table_len_x, table_len_y = table_len, table_len
    
    Height, Width = depth_map.shape[:2]
    #print(Height, Width)
    Height_num = Height//table_len_y
    Width_num = Width//table_len_x
    #print(Height_num, Width_num)
    
    # 縦型A4のCanvasを準備
    cv = canvas.Canvas('./media/pdf/'+str(image_id)+'.pdf', pagesize=portrait(A4))
    cv.setTitle('Instruction')
    # フォント登録
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
    #完成図の挿入
    image_path_full = './media/lego_img/'+str(image_id)+'.png'
    cv2.imwrite(image_path_full, color_map)
    sample_im_height = (Height*190/Width)
    cv.drawImage(image=image_path_full, x=10*mm, y=(150-sample_im_height/2)*mm, width=190*mm, height=sample_im_height*mm, mask='auto')
    #改ページ
    cv.showPage()
    
    Color_list = []
    for i, key in enumerate(COLOR_DIC):
        Color_list.append(COLOR_DIC[key])
    Color_len = len(Color_list)
    Color_len_syou = Color_len//table_len
    color_list_data = [[ j*table_len+i+1 for i in range(table_len)] for j in range(Color_len_syou)]
    if Color_len%table_len!=0:
        color_list_data += [[table_len*Color_len_syou+i+1 for i in range(Color_len%table_len)]+[" "]*(table_len-Color_len%table_len)]
    table_color = Table(color_list_data, colWidths=12*mm, rowHeights=12*mm)
    table_color.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'HeiseiKakuGo-W5', 15), # フォント
        ('GRID', (0, 0), (-1, -1), 1, colors.black),       # 罫線
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    for y in range(len(color_list_data)):
        for x in range(table_len):
            if y*table_len+x < Color_len:
                b,g,r = Color_list[y*table_len+x]
                bgr = int(b)+int(g)+int(r)
                if bgr >= 382:
                    wordColor = colors.black
                else:
                    wordColor = colors.white
                table_color.setStyle(TableStyle([
                    ('TEXTCOLOR', (x,y), (x,y), wordColor),
                    ('BACKGROUND', (x,y), (x,y), colors.Color(1.0/255*r,1.0/255*g,1.0/255*b, 1))
                ]))
            else:
                table_color.setStyle(TableStyle([
                    ('SPAN', (x, y) , (-1, y))
                ]))
                break
    
    for h in range(Height_num+1):
        table_len_x, table_len_y = table_len, table_len
        if h==Height_num:
            if Height%table_len!=0:
                table_len_y = Height % table_len
            else:
                continue
        for w in range(Width_num+1):
            if w==Width_num:
                if Width%table_len!=0:
                    table_len_x = Width % table_len
                else:
                    continue
            #data = [[" " for _ in range(table_len)] for __ in range(table_len)]
            data_color = min_color_num[table_len*h:table_len*h+table_len_y, table_len*w:table_len*w+table_len_x].tolist()
            data_depth = depth_map[table_len*h:table_len*h+table_len_y, table_len*w:table_len*w+table_len_x].tolist()
            data = list(map(lambda x,y: list(map(lambda a,b: str(int(a)+1)+str("×")+str(int(b)), x,y)), data_color, data_depth))
            table = Table(data, colWidths=12*mm, rowHeights=12*mm)
            # tableの装飾
            table.setStyle(TableStyle([                              
                ('FONT', (0, 0), (-1, -1), 'HeiseiKakuGo-W5', 8), # フォント
                ('GRID', (0, 0), (-1, -1), 1, colors.black),       # 罫線
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            for x in range(table_len_x):
                for y in range(table_len_y):
                    b,g,r = color_map[table_len*h+y, table_len*w+x]
                    bgr = int(b)+int(g)+int(r)
                    if bgr >= 382:
                        wordColor = colors.black
                    else:
                        wordColor = colors.white
                    table.setStyle(TableStyle([
                        ('TEXTCOLOR', (x,y), (x,y), wordColor),
                        ('BACKGROUND', (x,y), (x,y), colors.Color(1.0/255*r,1.0/255*g,1.0/255*b, 1))
                    ]))
            #Artのカラーテーブル
            table_y = 10 + (table_len-table_len_y)*12
            table.wrapOn(cv, 15*mm, table_y*mm) # table位置
            table.drawOn(cv, 15*mm, table_y*mm)
            #使用する色のカラーテーブル
            table_color.wrapOn(cv, 15*mm, 195*mm) # table位置
            table_color.drawOn(cv, 15*mm, 195*mm)
            #画像の挿入
            image_path = f'./media/pdf/pdf_image/{image_id}_{h}_{w}.png'
            sample_image = np.full_like(color_map,255)
            sample_image = cv2.copyMakeBorder(sample_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0,0,0))
            sample_image[table_len*h+1:table_len*h+table_len_y+1, table_len*w+1:table_len*w+table_len_x+1] = color_map[table_len*h:table_len*h+table_len_y, table_len*w:table_len*w+table_len_x]
            cv2.imwrite(image_path, sample_image)
            cv.drawImage(image=image_path, x=15*mm, y=240*mm, width=(min((Width*50/Height),88))*mm, height=50*mm, mask='auto')
            cv.drawImage(image=image_path_full, x=107*mm, y=240*mm, width=(min((Width*50/Height),88))*mm, height=50*mm, mask='auto')
            #改ページ
            cv.showPage()
            """
            plt.imshow(color_map[table_len*h:table_len*h+table_len, table_len*w:table_len*w+table_len,::-1])
            plt.show()
            """
    # 保存
    cv.save()
    for h in range(Height_num+1):
        for w in range(Width_num+1):
            image_path = f'./media/pdf/pdf_image/{image_id}_{h}_{w}.png'
            try:
                os.remove(image_path)
            except:
                continue
"""
元画像（RGB画像および深度マップを入力）
LEGOARTの幅resize_wを指定
ブロックの最大の高さblock_hを指定
使用するブロックの色の辞書
"""
def main(img_rgb, img_depth, resize_w, block_h, color_dic,image_id):
    Height, Width = img_rgb.shape[:2]
    ratio = resize_w/Width
    resize_rgb = cv2.resize(img_rgb, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    
    resize_depth = cv2.resize(img_depth, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    depth_post = posterization(resize_depth, block_h)
    
    LEGO_rgb,min_color_num  = change_coler(resize_rgb, color_dic)
    
    rgb_depth_array = create_rgb_depth_map(LEGO_rgb, depth_post)
    creat_instruction(LEGO_rgb, depth_post, min_color_num, color_dic,image_id)
    """create_csv_file(LEGO_rgb, depth_post)"""
    
    """LEGOブロックの色のみの画像、指定した高さの深度マップ、rgb_depth情報を含んだnumpy配列"""
    return LEGO_rgb, depth_post, rgb_depth_array
def color_select(color_list):
    COLOR_DIC = {'0 White': [242, 243, 242], '1 Tan': [111, 160, 176], '2 Light Green': [168, 217, 173], '3 Maersk Blue': [195, 146, 53], '4 Pink': [172, 151,252 ],
             '5 Nougat': [104, 145, 208], '6 Red': [9, 26, 201], '7 Blue': [191, 85, 0], '8 Yellow': [55, 205, 242], '9 Black': [29, 19, 5],
             '10 Green': [65, 120, 35], '11 Md,Green': [117, 196, 127], '12 Bt,Green': [65, 171, 88], '13 Dark Orange': [0, 85, 169],
             '14 Light Violet': [226, 202, 201], '15 Md,Blue': [219, 147, 90], '16 Md,Orange': [11, 167, 255], '17 Orange': [24, 138, 254],
             '18 Blue-Violet': [202, 116, 104], '19 Light Turquoise': [175, 165, 85], '20 Lime': [11, 233, 187], '21 Magenta': [118, 31, 144],
             '22 Sand Blue': [161, 116, 96], '23 Md,Nougat': [42, 112, 204], '24 Dark Tan': [115, 138, 149], '25 Dark Blue': [99, 52, 10],
             '26 Dark Green': [50, 70, 24], '27 Sand Green': [172, 188, 160], '28 Dark Red': [15, 14, 114], '29 Bt,Lt Orange': [61, 187, 248],
             '30 Reddish Brown': [18, 42, 88], '31 Light Bluish Gray': [169, 165, 160], '32 Dark Bluish Gray': [104, 110, 108],
             '33 Very Lt, Bluish Gray': [224, 227, 230], '34 Bt, Lt Blue': [233, 195, 159], '35 Dark Pink': [160, 112, 200],
             '36 Bright Pink': [200, 173, 228], '37 Bt,Lt Yellow': [58, 240, 255], '38 Dark Purple': [145, 54, 63], '39 Light Nougat': [179, 215, 246],
             '40 Dark Brown': [0, 33, 53], '41 Light Aqua': [234, 242, 211], '42 Md,Lavender': [185, 110, 160], '43 Lavender': [222, 164, 205],
             '44 Coral': [80, 127, 255]
            }
    new=dict()
    for i in color_list:
        new[i]=COLOR_DIC[i]
    return new
def d_hex(rgb_depth_array):
    new1=[]
    new2=[]
    for i in rgb_depth_array:
        new1_new=[]
        new2_new=[]
        for j in i:
            new1_new.append(hex(j[2]*256*256+j[1]*256+j[0]))
            new2_new.append(j[3])
        new1.append(new1_new)
        new2.append(new2_new)
    return [new1,new2]
def henkan(input_url,image_id, resize_w, block_h, color_list):
    image_original = cv2.imread("./"+input_url)
    depth_map = cv2.imread("./media/depth_img/"+str(image_id)+".png", 0)
    COLOR_DIC=color_select(color_list)
    lego_rgb, lego_depth, rgb_depth_array = main(image_original, depth_map, resize_w, block_h, COLOR_DIC,image_id)
    rgb_url="/media/lego_img/"+str(image_id)+".png"
    depth_url="/media/lego_depth/"+str(image_id)+".png"
    cv2.imwrite("."+rgb_url,lego_rgb)
    cv2.imwrite("."+depth_url,lego_depth)
    return lego_rgb,lego_depth,d_hex(rgb_depth_array)
