import os
import base64
import logging
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from doc_scanner.dsnt import dsnt

# 设置日志
logger = logging.getLogger(__name__)


class DocScanner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.graph = None
        self.session = None
        self.load_model()

    def load_model(self):
        """
        加载 TensorFlow 模型
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False

            self.graph = self.load_graph(self.model_path)

            # 获取输入输出张量
            self.inputs = self.graph.get_tensor_by_name('input:0')
            self.activation_map = self.graph.get_tensor_by_name("heats_map_regression/pred_keypoints/BiasAdd:0")

            # 构建 dsnt 操作
            with self.graph.as_default():
                self.hm1, self.hm2, self.hm3, self.hm4, self.kp1, self.kp2, self.kp3, self.kp4 = self.build_dsnt_operations(
                    self.activation_map)

            # 创建会话
            self.session = tf.compat.v1.Session(graph=self.graph)
            logger.info(f"TensorFlow模型加载成功: {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False

    def load_graph(self, frozen_graph_filename):
        """
        加载冻结的 TensorFlow 图
        """
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph

    def build_dsnt_operations(self, activation_map):
        """在图中构建 dsnt 操作"""
        hm1, kp1 = dsnt(activation_map[..., 0])
        hm2, kp2 = dsnt(activation_map[..., 1])
        hm3, kp3 = dsnt(activation_map[..., 2])
        hm4, kp4 = dsnt(activation_map[..., 3])
        return hm1, hm2, hm3, hm4, kp1, kp2, kp3, kp4

    def detect_document(self, image_path):
        """
        使用 TensorFlow 模型进行文档检测和关键点定位
        """
        try:
            # 读取并预处理图像
            img_nd = np.array(Image.open(image_path).resize((600, 800)))
            img_nd_bgr = cv2.cvtColor(img_nd, cv2.COLOR_RGB2BGR)

            # 运行模型
            hm1_nd, hm2_nd, hm3_nd, hm4_nd, kp1_nd, kp2_nd, kp3_nd, kp4_nd = self.session.run(
                [self.hm1, self.hm2, self.hm3, self.hm4, self.kp1, self.kp2, self.kp3, self.kp4],
                feed_dict={self.inputs: np.expand_dims(img_nd, 0)}
            )

            # 处理关键点
            keypoints_nd = np.array([kp1_nd[0], kp2_nd[0], kp3_nd[0], kp4_nd[0]])
            keypoints_nd = ((keypoints_nd + 1) / 2 * np.array([600, 800])).astype('int')

            # 计算新的角点
            x1 = (keypoints_nd[0, 0] + keypoints_nd[2, 0]) / 2.0
            y1 = (keypoints_nd[0, 1] + keypoints_nd[1, 1]) / 2.0
            x2 = (keypoints_nd[1, 0] + keypoints_nd[3, 0]) / 2.0
            y2 = (keypoints_nd[2, 1] + keypoints_nd[3, 1]) / 2.0

            new_kp1, new_kp2, new_kp3, new_kp4 = np.array([x1, y1]), np.array([x2, y1]), np.array([x1, y2]), np.array(
                [x2, y2])

            src_pts = keypoints_nd.astype('float32')
            dst_pts = np.array([new_kp1, new_kp2, new_kp3, new_kp4]).astype('float32')

            return img_nd_bgr, src_pts, dst_pts

        except Exception as e:
            logger.error(f"文档检测失败: {str(e)}")
            # 返回原始图像和None标记
            try:
                image = cv2.imread(image_path)
                return image, None, None
            except:
                raise e

    def perspective_correction(self, image, src_pts, dst_pts):
        """
        执行透视变换校正文档
        """
        if src_pts is None or dst_pts is None:
            return None

        try:
            # 计算变换矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 计算缩放因子和新尺寸
            x1, y1 = dst_pts[0]
            x2, y2 = dst_pts[3]
            resize_factor = image.shape[1] * 1.0 / (x2 - x1)
            height, width = image.shape[:2]
            new_height, new_width = int(height * resize_factor), int(width * resize_factor)

            # 执行透视变换
            transformed_image = cv2.warpPerspective(image, H, (new_width, new_height))

            # 裁剪文档区域
            cropped_image = transformed_image[int(y1):int(y2), int(x1):int(x2), :]

            return cropped_image

        except Exception as e:
            logger.error(f"透视校正失败: {str(e)}")
            return None

    def image_to_base64(self, image):
        """
        将OpenCV图像转换为base64字符串
        """
        if image is None:
            return None

        try:
            ret, buffer = cv2.imencode('.jpg', image)
            img_bytes = buffer.tobytes()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            return base64_str
        except Exception as e:
            logger.error(f"图像编码失败: {str(e)}")
            return None

    def scan_document(self, image_path):
        """
        主扫描函数 - 与DocumentScanner接口保持一致
        """
        try:
            # 1. 检测文档和关键点
            image, src_pts, dst_pts = self.detect_document(image_path)

            # 检查是否成功检测到文档关键点
            if src_pts is None or dst_pts is None:
                logger.info("未检测到文档关键点，返回原始图像")
                base64_str = self.image_to_base64(image)

                return {
                    "status": "success",
                    "message": "未检测到文档关键点，返回原始图像",
                    "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                    "document_detected": False
                }

            # 2. 执行透视校正
            corrected_image = self.perspective_correction(image, src_pts, dst_pts)

            if corrected_image is None:
                logger.warning("透视校正失败，返回原始图像")
                base64_str = self.image_to_base64(image)

                return {
                    "status": "success",
                    "message": "透视校正失败，返回原始图像",
                    "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                    "document_detected": False
                }

            # 3. 返回校正后的图像
            base64_str = self.image_to_base64(corrected_image)

            return {
                "status": "success",
                "message": "文档扫描完成",
                "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                "document_detected": True
            }

        except Exception as e:
            logger.error(f"文档扫描失败: {str(e)}")
            # 在异常情况下尝试返回原始图像
            try:
                image = cv2.imread(image_path)
                base64_str = self.image_to_base64(image)

                return {
                    "status": "error",
                    "message": f"文档扫描失败: {str(e)}，返回原始图像",
                    "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                    "document_detected": False
                }
            except:
                return {
                    "status": "error",
                    "message": f"文档扫描失败: {str(e)}",
                    "data": None,
                    "document_detected": False
                }

    def __del__(self):
        """
        析构函数，确保会话被关闭
        """
        if self.session:
            self.session.close()