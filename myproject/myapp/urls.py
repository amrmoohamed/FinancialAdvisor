from django.urls import path, include
from .views import ChatbotView

urlpatterns = [
    # ... other URLs ...
    path('chatbot/', ChatbotView.as_view(), name='chatbot'),
]