from email.policy import default
from turtle import color
from django import forms
from .models import UploadImage
from django.core.validators import MaxValueValidator, MinValueValidator
class UploadForm(forms.ModelForm):
    class Meta:
        model = UploadImage
        fields = ['image']
class SettingForm(forms.Form):
    takasa = forms.IntegerField(required=True,initial=5,label="高さ",validators=[MinValueValidator(1), MaxValueValidator(100)])
    haba = forms.IntegerField(required=True,initial=100,label="横のブロック数",validators=[MinValueValidator(1), MaxValueValidator(1000)])
    COLOR_DIC = {'0 White': [242, 243, 242], '1 Tan': [111, 160, 176], '2 Light Green': [168, 217, 173], '3 Maersk Blue': [195, 146, 53], '4 Pink': [255, 217, 171],
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
    colorkey=list(COLOR_DIC.keys())
    choices=[]
    val=[]
    for i in range(len(colorkey)):
        choices.append((colorkey[i], colorkey[i]))
        val.append(colorkey[i])
    colors = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple, label="Legoの色",choices=choices,required=True,initial=val)