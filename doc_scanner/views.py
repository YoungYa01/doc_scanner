from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
import logging
import os

from doc_scanner.DocumentScanner import DocumentScanner

# 设置日志
logger = logging.getLogger(__name__)

# 全局扫描器实例
scanner = None


def initialize_scanner():
    global scanner
    if scanner is None:
        model_path = getattr(settings, 'DOCUMENT_SCANNER_MODEL_PATH', None)
        if model_path:
            logger.info(f"正在加载模型: {model_path}")
        scanner = DocumentScanner(model_path)

        # 检查模型是否加载成功
        if scanner.model == "opencv_fallback":
            logger.warning("使用 OpenCV 备用方法，YOLO 模型未加载")


def index(request):
    """
    首页视图 - 返回前端页面
    """
    return render(request, 'index.html')


@csrf_exempt
@require_http_methods(["POST"])
def document_scan(request):
    """
    文档扫描API接口 - 返回识别后的图片
    """
    global scanner

    if scanner is None:
        initialize_scanner()

    if 'image' not in request.FILES:
        return JsonResponse({
            "status": "error",
            "message": "请上传图片文件",
            "data": None
        }, status=400)

    uploaded_file = request.FILES['image']

    # 检查文件大小（限制为10MB）
    max_size = 15 * 1024 * 1024  # 15MB
    if uploaded_file.size > max_size:
        return JsonResponse({
            "status": "error",
            "message": f"文件大小超过限制（最大15MB），当前文件大小为 {uploaded_file.size // 1024}KB",
            "data": None
        }, status=400)

    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if uploaded_file.content_type not in allowed_types:
        return JsonResponse({
            "status": "error",
            "message": "不支持的文件类型，请上传JPEG或PNG格式的图片",
            "data": None
        }, status=400)

    try:
        file_path = default_storage.save(f'uploads/{uploaded_file.name}', uploaded_file)
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        logger.info(f"文件保存成功: {full_file_path}, 大小: {uploaded_file.size} 字节")

        result = scanner.scan_document(full_file_path)

        # 清理临时文件
        if os.path.exists(full_file_path):
            os.remove(full_file_path)

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")

        if 'full_file_path' in locals() and os.path.exists(full_file_path):
            os.remove(full_file_path)

        return JsonResponse({
            "status": "error",
            "message": f"处理图片时发生错误: {str(e)}",
            "data": None
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """
    健康检查接口
    """
    global scanner
    if scanner is None:
        initialize_scanner()

    model_status = "loaded" if scanner.model and scanner.model != "opencv_fallback" else "fallback"

    return JsonResponse({
        "status": "success",
        "message": "服务正常运行",
        "model_status": model_status,
        "model_type": scanner.model_type if hasattr(scanner, 'model_type') else "unknown",
        "device": scanner.device if hasattr(scanner, 'device') else "unknown"
    })