from django.shortcuts import render
from rest_framework import viewsets

from cars.models import Vehicle
from cars.serializers import VehicleSerializer


class VehicleViewSet(viewsets.ModelViewSet):
    queryset = Vehicle.objects.all()
    serializer_class = VehicleSerializer
