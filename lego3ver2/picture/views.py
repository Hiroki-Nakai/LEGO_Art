from django.shortcuts import render, get_object_or_404
from .forms import UploadForm, SettingForm
from .models import UploadImage
from .gasyori import super_resolve
from .iro import henkan
import os
import json
from django.http import HttpResponse
COLOR_DIC ={'1 White': [242, 243, 242], '2 Tan': [111, 160, 176], '3 Light Green': [168, 217, 173], '4 Maersk Blue': [195, 146, 53], '5 Pink': [172, 151,252 ],
             '6 Nougat': [104, 145, 208], '7 Red': [9, 26, 201], '8 Blue': [191, 85, 0], '9 Yellow': [55, 205, 242], '10 Black': [29, 19, 5],
             '11 Green': [65, 120, 35], '12 Md,Green': [117, 196, 127], '13 Bt,Green': [65, 171, 88], '14 Dark Orange': [0, 85, 169],
             '15 Light Violet': [226, 202, 201], '16 Md,Blue': [219, 147, 90], '17 Md,Orange': [11, 167, 255], '18 Orange': [24, 138, 254],
             '19 Blue-Violet': [202, 116, 104], '20 Light Turquoise': [175, 165, 85], '21 Lime': [11, 233, 187], '22 Magenta': [118, 31, 144],
             '23 Sand Blue': [161, 116, 96], '24 Md,Nougat': [42, 112, 204], '25 Dark Tan': [115, 138, 149], '26 Dark Blue': [99, 52, 10],
             '27 Dark Green': [50, 70, 24], '28 Sand Green': [172, 188, 160], '29 Dark Red': [15, 14, 114], '30 Bt,Lt Orange': [61, 187, 248],
             '31 Reddish Brown': [18, 42, 88], '32 Light Bluish Gray': [169, 165, 160], '33 Dark Bluish Gray': [104, 110, 108],
             '34 Very Lt, Bluish Gray': [224, 227, 230], '35 Bt, Lt Blue': [233, 195, 159], '36 Dark Pink': [160, 112, 200],
             '37 Bright Pink': [200, 173, 228], '38 Bt,Lt Yellow': [58, 240, 255], '39 Dark Purple': [145, 54, 63], '40 Light Nougat': [179, 215, 246],
             '41 Dark Brown': [0, 33, 53], '42 Light Aqua': [234, 242, 211], '43 Md,Lavender': [185, 110, 160], '44 Lavender': [222, 164, 205],
             '45 Coral': [80, 127, 255]
        }
def d_hex(rgb):
        #print(rgb)
        a=format(int(rgb[2]), '02x')+format(int(rgb[1]), '02x')+format(int(rgb[0]), '02x')
        return "#"+a
def d_sirokuro(rgb):
        a=int(rgb[2])+int(rgb[1])+int(rgb[0])
        return "#000000"if a>382 else "#ffffff"
def index(request):
    params = {
        'title': '画像のアップロード',
        'upload_form': UploadForm(),
        'id': None,
    }

    if (request.method == 'POST'):
        form = UploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            upload_image = form.save()

            params['id'] = upload_image.id
            return preview(request,params["id"])
    return render(request, 'picture/index.html', params)
def preview(request, image_id=0):
    upload_image = get_object_or_404(UploadImage, id=image_id)
    if not os.path.isfile("./media/depth_img/"+str(image_id)+".png"):
        depth=super_resolve(upload_image.image.url,"./media/depth_img/"+str(image_id)+".png")
    colorkey=list(COLOR_DIC.keys())
    choices=[]
    val=[]
    for i in range(len(colorkey)):
        choices.append((colorkey[i], d_hex(COLOR_DIC[colorkey[i]]), d_sirokuro(COLOR_DIC[colorkey[i]])))
        val.append(colorkey[i])
    params = {
        'title': '画像の表示',
        'id': upload_image.id,
        'img': upload_image.image.url,
        'setting_form': SettingForm(),
        'colors': choices
    }

    return render(request, './picture/preview.html', params)
def transform(request, image_id=0):
    colorkey=list(COLOR_DIC.keys())
    choices=[]
    val=[]
    for i in range(len(colorkey)):
        choices.append((colorkey[i], d_hex(COLOR_DIC[colorkey[i]]), d_sirokuro(COLOR_DIC[colorkey[i]])))
    #print(choices)
    upload_image = get_object_or_404(UploadImage, id=image_id)
    if (request.method == 'POST'):
        form = SettingForm(request.POST)
        #print(request.POST)
        if form.is_valid():
            haba= form.cleaned_data.get('haba')
            takasa = form.cleaned_data.get('takasa')
            colors = request.POST.getlist('colors')
            #print(colors)
            rgb,depth,sekkei=henkan(upload_image.image.url,image_id,haba,takasa,colors)
            rgb_url="/media/lego_img/"+str(image_id)+".png"
            #print(sekkei)
            params = {
                'title': '画像処理',
                'id': upload_image.id,
                'setting_form': form,
                'img': upload_image.image.url,
                'depth': rgb_url,
                 'sekkei':sekkei,
                 'pdf':'./media/pdf/'+str(image_id)+'.pdf',
                 'colors':choices
            }

            return render(request, './picture/kakunin.html', params)


    params = {
        'title': '画像処理',
        'id': upload_image.id,
        'setting_form': SettingForm(),
        'img': upload_image.image.url,
        'result_url': '',
        'colors':val
    }

    return render(request, './picture/kakunin.html', params)
def hyouji(request):
    #print(request.POST)
    sekkei=eval(request.POST.get("sekkei"))
    #print(sekkei)
    data={"color":sekkei[0],"takasa":sekkei[1]}
    params={
        'data':json.dumps(data)
    }
    #print(params)
    return render(request,'./picture/3D.html',params)

def pdf(request,image_id=0):
    download_pth = './media/pdf/'+str(image_id)+'.pdf'
    download_name = 'sekkeizu.pdf'
    if os.path.exists(download_pth ):
        with open(download_pth , 'rb') as fh:
            response = HttpResponse(fh.read(), content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename*=UTF-8\'\'{}'.format(download_name)
    else:
        print("dame")
    return response