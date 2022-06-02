from django.db import models

# Create your models here.
class ml_model(models.Model):
    model=models.FileField(upload_to ='ml_models/')
    desc=models.CharField(max_length=20)

    def __str__(self):
        return self.desc
