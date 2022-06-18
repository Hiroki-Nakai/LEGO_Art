from django.conf.urls import url
from django.urls import path
from .import views

urlpatterns=[
    path('',views.index,name="index"),
    path('preview/<int:image_id>/', views.preview, name='preview'),
    path('transform/<int:image_id>/', views.transform, name='transform'),
    path('hyouji/', views.hyouji, name='hyouji'),
    path('pdf/<int:image_id>', views.pdf, name='pdf'),

]