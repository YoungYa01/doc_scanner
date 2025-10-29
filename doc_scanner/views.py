import base64
import json

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
import logging
import os

from doc_scanner.DocumentScanner import DocumentScanner
from doc_scanner.utils import process_image_base, process_grayscale, process_sharpen, process_black_white, \
    process_enhance

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

    # 检查文件大小（限制为15MB）
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
@require_http_methods(["POST"])
def process_grayscale_api(request):
    """接口1：灰度处理 - 支持参数"""
    try:
        # 获取图像数据和参数
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            image_str = data.get('image', '')
            params = data.get('params', {})
        else:
            if 'image' not in request.FILES:
                return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)
            uploaded_file = request.FILES['image']
            image_data = uploaded_file.read()
            image_str = base64.b64encode(image_data).decode('utf-8')
            # 对于表单数据，可以尝试从请求中获取参数
            params = {
                'method': request.POST.get('method', 'average'),
            }

        if not image_str:
            return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)

        # 解码 base64 图像
        if ',' in image_str:
            image_str = image_str.split(',')[1]

        image_data = base64.b64decode(image_str)

        # 处理图像
        result = process_image_base(image_data, process_grayscale, "grayscale", **params)

        if result['success']:
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=500)

    except Exception as e:
        logger.error(f"灰度处理接口错误: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_sharpen_api(request):
    """接口2：锐化处理 - 支持参数"""
    try:
        # 获取图像数据和参数
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            image_str = data.get('image', '')
            params = data.get('params', {})
        else:
            if 'image' not in request.FILES:
                return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)
            uploaded_file = request.FILES['image']
            image_data = uploaded_file.read()
            image_str = base64.b64encode(image_data).decode('utf-8')
            params = {
                'intensity': float(request.POST.get('intensity', 1.0)),
                'use_unsharp_mask': request.POST.get('use_unsharp_mask', 'false').lower() == 'true',
                'sigma': float(request.POST.get('sigma', 1.0)),
            }

        if not image_str:
            return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)

        # 解码 base64 图像
        if ',' in image_str:
            image_str = image_str.split(',')[1]

        image_data = base64.b64decode(image_str)

        # 处理图像
        result = process_image_base(image_data, process_sharpen, "sharpen", **params)

        if result['success']:
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=500)

    except Exception as e:
        logger.error(f"锐化处理接口错误: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_black_white_api(request):
    """接口3：黑白处理 - 支持参数"""
    try:
        # 获取图像数据和参数
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            image_str = data.get('image', '')
            params = data.get('params', {})
        else:
            if 'image' not in request.FILES:
                return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)
            uploaded_file = request.FILES['image']
            image_data = uploaded_file.read()
            image_str = base64.b64encode(image_data).decode('utf-8')
            params = {
                'method': request.POST.get('method', 'adaptive'),
                'threshold': int(request.POST.get('threshold', 127)),
                'block_size': int(request.POST.get('block_size', 11)),
                'c': int(request.POST.get('c', 2)),
                'apply_morphology': request.POST.get('apply_morphology', 'false').lower() == 'true',
                'kernel_size': int(request.POST.get('kernel_size', 3)),
            }

        if not image_str:
            return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)

        # 解码 base64 图像
        if ',' in image_str:
            image_str = image_str.split(',')[1]

        image_data = base64.b64decode(image_str)

        # 处理图像
        result = process_image_base(image_data, process_black_white, "black_white", **params)

        if result['success']:
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=500)

    except Exception as e:
        logger.error(f"黑白处理接口错误: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def process_enhance_api(request):
    """接口4：增强处理 - 支持参数"""
    try:
        # 获取图像数据和参数
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            image_str = data.get('image', '')
            params = data.get('params', {})
        else:
            if 'image' not in request.FILES:
                return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)
            uploaded_file = request.FILES['image']
            image_data = uploaded_file.read()
            image_str = base64.b64encode(image_data).decode('utf-8')
            params = {
                'clip_limit': float(request.POST.get('clip_limit', 3.0)),
                'tile_grid_size': int(request.POST.get('tile_grid_size', 8)),
                'brightness': float(request.POST.get('brightness', 1.0)),
                'contrast': float(request.POST.get('contrast', 1.0)),
                'sharpen_after_enhance': request.POST.get('sharpen_after_enhance', 'false').lower() == 'true',
            }

        if not image_str:
            return JsonResponse({'success': False, 'error': '未提供图像数据'}, status=400)

        # 解码 base64 图像
        if ',' in image_str:
            image_str = image_str.split(',')[1]

        image_data = base64.b64decode(image_str)

        # 处理图像
        result = process_image_base(image_data, process_enhance, "enhance", **params)

        if result['success']:
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=500)

    except Exception as e:
        logger.error(f"增强处理接口错误: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
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