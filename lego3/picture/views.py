from django.shortcuts import render, get_object_or_404
from .forms import UploadForm, SettingForm
from .models import UploadImage
from .gasyori import super_resolve
from .iro import henkan
import os
import json
from django.http import HttpResponse

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
    params = {
        'title': '画像の表示',
        'id': upload_image.id,
        'img': upload_image.image.url,
        'setting_form': SettingForm(),
    }

    return render(request, './picture/preview.html', params)
def transform(request, image_id=0):

    upload_image = get_object_or_404(UploadImage, id=image_id)
    if (request.method == 'POST'):
        form = SettingForm(request.POST)
        #print(request.POST)
        if form.is_valid():
            
            haba= form.cleaned_data.get('haba')
            takasa = form.cleaned_data.get('takasa')
            colors = form.cleaned_data.get('colors')
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
                 'pdf':'./media/pdf/'+str(image_id)+'.pdf'
            }

            return render(request, './picture/kakunin.html', params)


    params = {
        'title': '画像処理',
        'id': upload_image.id,
        'setting_form': SettingForm(),
        'img': upload_image.image.url,
        'result_url': ''
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