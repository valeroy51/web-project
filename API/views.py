# from django.shortcuts import render
# from .models import Station
# from rest_framework_simplejwt.views import TokenObtainPairView
# from .jwt import MyTokenObtainPairSerializer
# from rest_framework.decorators import api_view, permission_classes
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.response import Response

# class MyTokenObtainPairView(TokenObtainPairView):
#     serializer_class = MyTokenObtainPairSerializer


# @api_view(["GET"])
# @permission_classes([IsAuthenticated])
# def api_me(request):
#     u = request.user
#     return Response({
#         "id": u.id,
#         "username": u.username,
#         "is_staff": u.is_staff,
#         "is_authenticated": u.is_authenticated,
#     })