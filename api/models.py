from django.db import models

# Create your models here.
class ml_model(models.Model):
    model=models.FileField(upload_to ='ml_models/')
    desc=models.CharField(max_length=20)

    def __str__(self):
        return self.desc

class ModelPredictions(models.Model):
    
    pred_type=models.TextField(max_length=100,blank=True)  #twitter , news
    time=models.DateTimeField(auto_now=True)         # time (when you hit the query..)
    query_time=models.DateTimeField(auto_now=False)         # time (when you hit the query..)
    positive_count = models.TextField(max_length=5)  # count of positive pred
    negetive_count = models.TextField(max_length=5)   # count of negetive pred
    neutral_count = models.TextField(max_length=5)   # count of neutral pred
    total_result = models.TextField(max_length=5)   # count of total rows.
    query_String = models.TextField(max_length=50)   # count of total rows.
    location_cordinate = models.TextField(max_length=100)   # location co-ordinate (actual coordinate)
    location_string = models.TextField(max_length=100)   # location String (new delhi,kashmeer)
    

    
    def __str__(self):
        return "{}-{}-{}".format(self.query_String,self.id,self.query_time)
class newsModelPredictions(models.Model):
    
    pred_type=models.TextField(max_length=100,blank=True)  #twitter , news
    time=models.DateTimeField(auto_now=True)         # time (when you hit the query..)
    positive_count = models.TextField(max_length=5)  # count of positive pred
    negetive_count = models.TextField(max_length=5)   # count of negetive pred
    neutral_count = models.TextField(max_length=5)   # count of neutral pred
    total_result = models.TextField(max_length=5)   # count of total rows.
    query_String = models.TextField(max_length=50)   # count of total rows.
    # location_cordinate = models.TextField(max_length=100)   # location co-ordinate (actual coordinate)
    source = models.TextField(max_length=100)   # (source of news)
    

    
    def __str__(self):
        return "{}-{}-{}".format(self.query_String,self.id,self.time)