from django.urls import path, include

from cars.router import router

urlpatterns = [
    path('', include(router.urls))
]
