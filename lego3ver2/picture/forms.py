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