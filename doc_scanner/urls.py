"""
URL configuration for doc_scanner project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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

from django.urls import path, include

from doc_scanner import views

urlpatterns = [
    path('example', views.index, name='example'),
    path('doc_scan/scan', views.document_scan, name='scan_document'),
    path('doc_scan/scanner', views.document_scan_tf, name='scan_document'),
    path('doc_scan/grayscale', views.process_grayscale_api, name='grayscale'),
    path('doc_scan/sharpen', views.process_sharpen_api, name='sharpen'),
    path('doc_scan/black-white', views.process_black_white_api, name='black_white'),
    path('doc_scan/enhance', views.process_enhance_api, name='enhance'),
    path('doc_scan/health', views.health_check, name='health_check'),
]
