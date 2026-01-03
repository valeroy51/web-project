"""
URL configuration for Skripsi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from . import views
from API import views as api_views
from django.contrib.auth import logout
from django.contrib.auth.views import LogoutView
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('', views.home, name='home'),
    path('tentang/', views.tentang, name='tentang'),
    path('input/', views.data, name='input'),
    path('analisis/', views.analisis, name='analisis'),
    path('prediksi/<int:id_station>/', views.prediksi, name='prediksi'),
    path("informasi-polutan/", views.polutan_info, name="polutan_info"),    
    
    path("admin-login/", views.admin_login, name="admin_login"),
    path("logout/", views.admin_logout, name="logout"),
    
    path('merge/start/', views.merge, name='merge'),
    path("training/start/", views.train, name="train"),
    path("schedule-train/", views.schedule_train, name="schedule_train"),
    
    path("api/auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]

if settings.DEBUG:
    # Include django_browser_reload URLs only in DEBUG mode
    urlpatterns += [
        path("__reload__/", include("django_browser_reload.urls")),
    ]