from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse, reverse_lazy
import joblib
from sklearn.preprocessing import LabelEncoder
from .utility import image_predicition
from tensorflow.keras.models import load_model


Pneumonia_MODEL = load_model('PneumoniaPredict/pnem.h5')



class AboutWebsite(TemplateView):
    template_name = "PneumoniaPredict/about.html"


class ContactWebsite(TemplateView):
    template_name = "PneumoniaPredict/contact.html"


class HomeWebsite(TemplateView):
    template_name = "PneumoniaPredict/index.html"


def PredictionPage(request):
    output = ''
    if request.method == "POST":
            output = image_predicition(request.FILES['image'], Pneumonia_MODEL)
            return render(request, "PneumoniaPredict/result.html",
                          {"output": output})

    else:
        output=''
    return render(request, "PneumoniaPredict/predict.html",
                  {"output": output})
