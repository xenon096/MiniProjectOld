from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class detection_type(models.Model):

    SV_TAXONOMY= models.CharField(max_length=30000)
    TAXONOMY_NAME= models.CharField(max_length=30000)
    RG_REFERENCE= models.CharField(max_length=30000)
    SV_REFERENCE= models.CharField(max_length=30000)
    SV_NAME= models.CharField(max_length=30000)
    SV_DESCRIPTION= models.CharField(max_length=30000)
    SL_REFERENCE= models.CharField(max_length=30000)
    LC_REFERENCE= models.CharField(max_length=30000)
    PHONE_NUMBER= models.CharField(max_length=30000)
    WEBSITE= models.CharField(max_length=30000)
    EMAIL_ADDRESS= models.CharField(max_length=30000)
    WHEELCHAIR_ACCESSIBLE= models.CharField(max_length=30000)
    STREET_NUMBER= models.CharField(max_length=30000)
    CITY= models.CharField(max_length=30000)
    LATITUDE= models.CharField(max_length=30000)
    LONGITUDE= models.CharField(max_length=30000)
    LINK_811= models.CharField(max_length=30000)
    Prediction= models.CharField(max_length=30000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



