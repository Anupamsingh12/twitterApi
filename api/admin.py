from django.contrib import admin
from .models import ml_model,ModelPredictions,newsModelPredictions
# Register your models here.
admin.site.register(ml_model)
admin.site.register(ModelPredictions)
admin.site.register(newsModelPredictions)
