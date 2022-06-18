from django.db import models

class UploadImage(models.Model):
    image = models.ImageField(upload_to='img/')
    depth_image = models.ImageField(upload_to='depth_img/',null=True)
    color_image = models.ImageField(upload_to='color_img/',null=True)