from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.conf import settings
import logging
import os
import cv2
import numpy as np
import base64

# 设置日志
logger = logging.getLogger(__name__)


class DocumentScanner:
    def __init__(self, model_path=None):
        """
        初始化文档扫描模型
        """
        self.model = None
        self.model_type = None
        self.device = 'cpu'  # 默认使用 CPU
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        加载 YOLO 模型
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                self.model = "opencv_fallback"
                return

            # 尝试导入 ultralytics (YOLOv8/v11)
            try:
                from ultralytics import YOLO
                import torch

                # 检查 CUDA 是否可用
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("CUDA 可用，使用 GPU 进行推理")
                else:
                    self.device = 'cpu'
                    logger.info("CUDA 不可用，使用 CPU 进行推理")

                # 加载模型并指定设备
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.model_type = "yolo"
                logger.info(f"YOLO模型加载成功: {model_path}，使用设备: {self.device}")

            except ImportError:
                logger.error("未安装 ultralytics 库，请运行: pip install ultralytics")
                self.model = "opencv_fallback"
                return
            except Exception as e:
                logger.error(f"YOLO模型加载失败: {str(e)}")
                # 如果 GPU 版本失败，尝试 CPU 版本
                try:
                    logger.info("尝试使用 CPU 加载模型...")
                    from ultralytics import YOLO
                    self.model = YOLO(model_path)
                    self.model.to('cpu')
                    self.model_type = "yolo"
                    self.device = 'cpu'
                    logger.info(f"YOLO模型在 CPU 上加载成功: {model_path}")
                except Exception as e2:
                    logger.error(f"CPU 模型加载也失败: {str(e2)}")
                    self.model = "opencv_fallback"
                return

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self.model = "opencv_fallback"

    def preprocess_image(self, image_path):
        """
        图像预处理
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("无法读取图像文件")
            return image
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise

    def detect_document(self, image):
        """
        使用 YOLO 模型进行文档检测
        """
        try:
            if self.model_type == "yolo":
                return self.detect_document_yolo(image)
            else:
                return self.detect_document_opencv(image)
        except Exception as e:
            logger.error(f"文档检测失败: {str(e)}")
            raise

    def detect_document_yolo(self, image):
        """
        使用 YOLO 模型进行文档检测
        """
        try:
            # 使用 YOLO 模型进行预测
            results = self.model(image, device=self.device)

            # 处理检测结果
            document_detected = False
            boxes = []
            masks = []

            # 创建结果图像的副本
            result_image = image.copy()

            for result in results:
                # 获取边界框
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        boxes.append({
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "confidence": float(conf),
                            "class": cls
                        })
                        document_detected = True

                        # 在图像上绘制检测框
                        cv2.rectangle(result_image,
                                      (int(x1), int(y1)),
                                      (int(x2), int(y2)),
                                      (0, 255, 0), 2)

                        # 添加置信度标签
                        label = f"Doc: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(result_image,
                                      (int(x1), int(y1) - label_size[1] - 10),
                                      (int(x1) + label_size[0], int(y1)),
                                      (0, 255, 0), -1)
                        cv2.putText(result_image, label,
                                    (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # 获取分割掩码（如果模型支持分割）
                if hasattr(result, 'masks') and result.masks is not None:
                    for mask in result.masks:
                        # 将掩码点转换为整数坐标
                        mask_points = np.array(mask.xy[0], dtype=np.int32)
                        masks.append(mask_points.tolist())

                        # 在图像上绘制分割轮廓
                        cv2.polylines(result_image, [mask_points], True, (255, 0, 0), 2)

            # 将结果图像编码为Base64
            _, buffer = cv2.imencode('.jpg', result_image)
            result_image_base64 = base64.b64encode(buffer).decode('utf-8')

            result_data = {
                "success": True,
                "document_detected": document_detected,
                "detections": boxes,
                "masks": masks,
                "result_image": result_image_base64,
                "model_type": "yolo",
                "device": self.device,
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }

            return result_data
        except Exception as e:
            logger.error(f"YOLO 检测失败: {str(e)}")
            # 如果 YOLO 失败，回退到 OpenCV
            return self.detect_document_opencv(image)

    def detect_document_opencv(self, image):
        """
        使用OpenCV进行文档检测（备用方法）
        """
        try:
            # 创建结果图像的副本
            result_image = image.copy()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)

            # 查找轮廓
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            # 寻找文档轮廓
            document_contour = None
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    document_contour = approx
                    break

            # 在图像上绘制检测到的轮廓
            if document_contour is not None:
                cv2.drawContours(result_image, [document_contour], -1, (0, 255, 0), 3)

                # 添加标签
                cv2.putText(result_image, "Document Detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 将结果图像编码为Base64
            _, buffer = cv2.imencode('.jpg', result_image)
            result_image_base64 = base64.b64encode(buffer).decode('utf-8')

            result = {
                "success": True,
                "document_detected": document_contour is not None,
                "result_image": result_image_base64,
                "model_type": "opencv_fallback",
                "device": "cpu",
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }

            return result
        except Exception as e:
            logger.error(f"OpenCV 检测失败: {str(e)}")
            raise

    def scan_document(self, image_path):
        """
        主扫描函数
        """
        try:
            image = self.preprocess_image(image_path)
            result = self.detect_document(image)

            return {
                "status": "success",
                "message": "文档扫描完成",
                "data": result
            }

        except Exception as e:
            logger.error(f"文档扫描失败: {str(e)}")
            return {
                "status": "error",
                "message": f"文档扫描失败: {str(e)}",
                "data": None
            }


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
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return JsonResponse({
            "status": "error",
            "message": f"文件大小超过限制（最大10MB），当前文件大小为 {uploaded_file.size // 1024}KB",
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