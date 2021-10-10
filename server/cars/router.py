from rest_framework import routers
from .views import VehicleViewSet

router = routers.DefaultRouter()
router.register('vehicle', VehicleViewSet)
