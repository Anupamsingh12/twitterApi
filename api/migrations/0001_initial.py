# Generated by Django 3.1.2 on 2021-04-15 03:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ml_model',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.FileField(upload_to='ml_models/')),
                ('desc', models.CharField(max_length=20)),
            ],
        ),
    ]
