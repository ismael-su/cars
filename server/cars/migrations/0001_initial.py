# Generated by Django 3.2.7 on 2021-09-17 14:07

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Vehicle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vehicle_id', models.CharField(max_length=50)),
                ('vehicle_class', models.CharField(max_length=50)),
                ('accuracy', models.CharField(max_length=50)),
                ('speed', models.CharField(max_length=50)),
                ('position', models.CharField(max_length=50)),
                ('state', models.CharField(max_length=50)),
                ('time', models.CharField(max_length=50)),
            ],
        ),
    ]
