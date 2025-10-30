import base64
import logging
import os

import cv2
import numpy as np

import torch

# 设置日志
logger = logging.getLogger(__name__)


class DocumentScanner:
    def __init__(self, model_path=None):
        """
        初始化文档扫描模型
        """
        self.model = None
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
                    self.device = 'cpu'
                    logger.info(f"YOLO模型在 CPU 上加载成功: {model_path}")
                except Exception as e2:
                    logger.error(f"CPU 模型加载也失败: {str(e2)}")
                    self.model = "opencv_fallback"
                return

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self.model = "opencv_fallback"

    def detect_document(self, image_path):
        """
        使用 YOLO 模型进行文档检测
        """
        try:
            return self.detect_document_yolo(image_path)
        except Exception as e:
            logger.error(f"文档检测失败: {str(e)}")
            raise

    def detect_document_yolo(self, image_path):
        """
        使用 YOLO 模型进行文档检测
        """
        try:
            # 使用 YOLO 模型进行预测
            results = self.model.predict(image_path, device=self.device)
            result = results[0]
            mask = None

            # 读取原始图像
            image = cv2.imread(image_path)

            # 提取文档 mask（假设只有一个文档对象）
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    name = result.names[cls_id]
                    if name == 'document' or len(result.masks.data) > 0:
                        mask = result.masks.data[0].cpu().numpy()
                        break

            if mask is None:
                logger.warning("未检测到文档区域，返回原始图像")
                # 返回原始图像和None标记
                return image, None

            # 转为 OpenCV 格式二值图
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
            return image, mask

        except Exception as e:
            logger.error(f"YOLO 检测失败: {str(e)}")
            # 如果 YOLO 失败，尝试返回原始图像
            try:
                image = cv2.imread(image_path)
                return image, None
            except:
                return "Error!" + str(e)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_document_corners(self, mask):
        if mask is None:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect)
            # 修复：使用 np.int32 替代已弃用的 np.int0
            corners = np.int32(corners)

        return self.order_points(corners)

    def perspective_correction(self, image, corners):
        if corners is None:
            # 如果没有角点，直接返回原始图像
            ret, buffer = cv2.imencode('.jpg', image)
            img_bytes = buffer.tobytes()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            return base64_str

        width = int(max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])
        ))
        height = int(max(
            np.linalg.norm(corners[0] - corners[3]),
            np.linalg.norm(corners[1] - corners[2])
        ))

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        ret, buffer = cv2.imencode('.jpg', warped)
        img_bytes = buffer.tobytes()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str

    def scan_document(self, image_path):
        """
        主扫描函数
        """
        try:
            # 1. 检测文档
            image, mask = self.detect_document(image_path)

            # 检查是否成功检测到文档
            if mask is None:
                logger.info("未检测到文档区域，返回原始图像")
                # 直接返回原始图像
                ret, buffer = cv2.imencode('.jpg', image)
                img_bytes = buffer.tobytes()
                base64_str = base64.b64encode(img_bytes).decode('utf-8')

                return {
                    "status": "success",
                    "message": "未检测到文档区域，返回原始图像",
                    "data": f"data:image/jpeg;base64,{base64_str}",
                    "document_detected": False
                }

            # 2. 提取角点并透视校正
            corners = self.get_document_corners(mask)
            result = self.perspective_correction(image, corners)

            return {
                "status": "success",
                "message": "文档扫描完成",
                "data": f"data:image/jpeg;base64,{result}",
                "document_detected": True
            }

        except Exception as e:
            logger.error(f"文档扫描失败: {str(e)}")
            # 即使在异常情况下，也尝试返回原始图像
            try:
                image = cv2.imread(image_path)
                ret, buffer = cv2.imencode('.jpg', image)
                img_bytes = buffer.tobytes()
                base64_str = base64.b64encode(img_bytes).decode('utf-8')

                return {
                    "status": "error",
                    "message": f"文档扫描失败: {str(e)}，返回原始图像",
                    "data": base64_str,
                    "document_detected": False
                }
            except:
                return {
                    "status": "error",
                    "message": f"文档扫描失败: {str(e)}",
                    "data": None,
                    "document_detected": False
                }