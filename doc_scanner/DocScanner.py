import os
import base64
import logging
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from doc_scanner.dsnt import dsnt

logger = logging.getLogger(__name__)

def order_points_clockwise_from_tr(pts):
    """
    将四个角点按顺时针排列，并且从右上角开始
    :param pts: shape (4,2)
    :return: 排序后的角点 np.array([[tr],[br],[bl],[tl]])
    """
    pts = np.array(pts, dtype=np.float32)
    center = np.mean(pts, axis=0)

    # 按顺时针角度排序
    def angle_from_center(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])

    angles = [angle_from_center(p) for p in pts]
    pts_angles = list(zip(pts, angles))
    pts_angles_sorted = sorted(pts_angles, key=lambda x: x[1])

    sorted_pts = np.array([p[0] for p in pts_angles_sorted])

    # 找到最靠近右上角的点作为起点
    tr_index = np.argmin(np.linalg.norm(sorted_pts - np.array([np.max(pts[:,0]), np.min(pts[:,1])]), axis=1))
    sorted_pts = np.roll(sorted_pts, -tr_index, axis=0)

    return sorted_pts

class DocScanner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.graph = None
        self.session = None
        self.load_model()

    # ---------- 模型加载 ----------
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False

            self.graph = self.load_graph(self.model_path)
            self.inputs = self.graph.get_tensor_by_name('input:0')
            self.activation_map = self.graph.get_tensor_by_name("heats_map_regression/pred_keypoints/BiasAdd:0")

            with self.graph.as_default():
                self.hm1, self.hm2, self.hm3, self.hm4, self.kp1, self.kp2, self.kp3, self.kp4 = self.build_dsnt_operations(
                    self.activation_map
                )

            self.session = tf.compat.v1.Session(graph=self.graph)
            logger.info(f"TensorFlow模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False

    def load_graph(self, frozen_graph_filename):
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph

    def build_dsnt_operations(self, activation_map):
        hm1, kp1 = dsnt(activation_map[..., 0])
        hm2, kp2 = dsnt(activation_map[..., 1])
        hm3, kp3 = dsnt(activation_map[..., 2])
        hm4, kp4 = dsnt(activation_map[..., 3])
        return hm1, hm2, hm3, hm4, kp1, kp2, kp3, kp4

    # ---------- 文档检测 ----------
    def detect_document(self, image_path):
        try:
            # 读取原图
            orig_img = cv2.imread(image_path)
            h_orig, w_orig = orig_img.shape[:2]

            # 模型输入 resize（保持比例）
            input_h, input_w = 800, 600
            img_resized = cv2.resize(orig_img, (input_w, input_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # 模型推理
            hm1_nd, hm2_nd, hm3_nd, hm4_nd, kp1_nd, kp2_nd, kp3_nd, kp4_nd = self.session.run(
                [self.hm1, self.hm2, self.hm3, self.hm4, self.kp1, self.kp2, self.kp3, self.kp4],
                feed_dict={self.inputs: np.expand_dims(img_rgb, 0)}
            )

            # 输出角点（0~1 归一化） → 映射回原图尺寸
            keypoints = np.array([kp1_nd[0], kp2_nd[0], kp3_nd[0], kp4_nd[0]])  # [-1,1]
            keypoints = (keypoints + 1) / 2  # [0,1] 相对坐标
            keypoints[:, 0] *= w_orig
            keypoints[:, 1] *= h_orig

            src_pts = order_points_clockwise_from_tr(keypoints)
            return orig_img, src_pts

        except Exception as e:
            logger.error(f"文档检测失败: {str(e)}")
            try:
                image = cv2.imread(image_path)
                return image, None
            except:
                raise e

    # ---------- 透视校正 ----------
    @staticmethod
    def perspective_correction(image, src_pts):
        if src_pts is None:
            return None

        widthA = np.linalg.norm(src_pts[1] - src_pts[0])
        widthB = np.linalg.norm(src_pts[2] - src_pts[3])
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(src_pts[2] - src_pts[1])
        heightB = np.linalg.norm(src_pts[3] - src_pts[0])
        maxHeight = int(max(heightA, heightB))

        dst_pts = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    # ---------- 图像编码 ----------
    @staticmethod
    def image_to_base64(image):
        if image is None:
            return None
        try:
            ret, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer.tobytes()).decode('utf-8')
        except Exception as e:
            logger.error(f"图像编码失败: {str(e)}")
            return None


    # ---------- 主扫描流程 ----------
    def scan_document(self, image_path):
        try:
            image, src_pts = self.detect_document(image_path)
            if src_pts is None:
                base64_str = self.image_to_base64(image)
                return {
                    "status": "success",
                    "message": "未检测到文档关键点，返回原始图像",
                    "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                    "document_detected": False
                }

            corrected_image = self.perspective_correction(image, src_pts)
            if corrected_image is None:
                base64_str = self.image_to_base64(image)
                return {
                    "status": "success",
                    "message": "透视校正失败，返回原始图像",
                    "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                    "document_detected": False
                }

            # ✅ 顺时针旋转 90 度
            corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)

            base64_str = self.image_to_base64(corrected_image)
            return {
                "status": "success",
                "message": "文档扫描完成",
                "data": f"data:image/jpeg;base64,{base64_str}" if base64_str else None,
                "document_detected": True
            }

        except Exception as e:
            logger.error(f"文档扫描失败: {str(e)}")
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

    # ---------- 析构 ----------
    def __del__(self):
        if self.session:
            self.session.close()