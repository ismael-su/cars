from django.db import models

class Vehicle(models.Model):
    vehicle_id = models.CharField(max_length=50)
    vehicle_class = models.CharField(max_length=50)
    accuracy = models.CharField(max_length=50)
    speed = models.CharField(max_length=50)
    position = models.CharField(max_length=50)
    state = models.CharField(max_length=50)
    time = models.CharField(max_length=50)

    def __str__(self):
        return self.vehicle_id
