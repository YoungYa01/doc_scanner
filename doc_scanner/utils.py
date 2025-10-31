import base64
import logging
import time
import io

import cv2
import numpy as np
from PIL import Image
import PIL.ExifTags

# 设置日志
logger = logging.getLogger(__name__)


def process_grayscale(image, **kwargs):
    """灰度处理 - 加权平均法"""
    try:
        # 使用加权平均法进行灰度化（ITU-R 601-2亮度变换）
        weights = [0.299, 0.587, 0.114]
        gray_image = np.dot(image[..., :3], weights)
        gray_image = gray_image.astype(np.uint8)

        # 将单通道灰度图转换为三通道以便显示
        gray_image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        return gray_image_bgr
    except Exception as e:
        logger.error(f"灰度处理失败: {str(e)}")
        raise


def process_sharpen(image, **kwargs):
    """锐化处理 - USM锐化算法"""
    try:
        # USM (Unsharp Mask)锐化算法
        # 1. 对原图进行高斯模糊
        # 2. 用原图减去模糊图得到细节图
        # 3. 将细节图按比例加到原图上
        
        # 使用固定参数进行高斯模糊
        sigma = 1.0
        # 计算合适的核大小，通常为sigma的6倍左右，且为奇数
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # 执行高斯模糊
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # 设置锐化强度（固定值）
        intensity = 1.0
        
        # 应用USM锐化：原图 + 强度*(原图 - 模糊图)
        sharpened = cv2.addWeighted(image, 1.0 + intensity, blurred, -intensity, 0)
        
        return sharpened
    except Exception as e:
        logger.error(f"锐化处理失败: {str(e)}")
        raise


def process_black_white(image, **kwargs):
    """黑白处理 - 二值化算法"""
    try:
        # 先转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用大津算法进行自适应阈值二值化
        # 大津算法会自动计算最优阈值，对不同光照条件下的文档图像处理效果较好
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 将单通道二值图转换为三通道以便显示
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary_bgr
    except Exception as e:
        logger.error(f"黑白处理失败: {str(e)}")
        raise


def process_enhance(image, **kwargs):
    """图像增强处理 - 增加参数控制"""
    try:
        # 对比度增强参数
        clip_limit = kwargs.get('clip_limit', 3.0)
        tile_grid_size = kwargs.get('tile_grid_size', 8)
        brightness = kwargs.get('brightness', 1.0)
        contrast = kwargs.get('contrast', 1.0)

        # 亮度对比度调整
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1.0) * 255)

        # 应用 CLAHE 对比度限制自适应直方图均衡化
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        l_enhanced = clahe.apply(l)

        # 合并通道并转换回 BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 可选：锐化增强后的图像
        if kwargs.get('sharpen_after_enhance', False):
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced
    except Exception as e:
        logger.error(f"图像增强失败: {str(e)}")
        raise


def image_to_base64(image):
    """将 OpenCV 图像转换为 base64 字符串 - 类似您原有代码"""
    try:
        success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            return base64.b64encode(buffer).decode('utf-8')
        return ""
    except Exception as e:
        logger.error(f"图像编码错误: {e}")
        return ""


def get_image_orientation(image_data):
    """获取图像的EXIF方向信息"""
    try:
        img = Image.open(io.BytesIO(image_data))
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                if tag in PIL.ExifTags.TAGS and PIL.ExifTags.TAGS[tag] == 'Orientation':
                    return value
    except:
        pass
    return 1  # 默认方向


def apply_orientation(image, orientation):
    """根据EXIF方向信息旋转图像"""
    if orientation == 1:
        return image
    elif orientation == 2:
        return cv2.flip(image, 1)  # 水平翻转
    elif orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)  # 旋转180度
    elif orientation == 4:
        return cv2.flip(image, 0)  # 垂直翻转
    elif orientation == 5:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(image, 1)
    elif orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 旋转90度
    elif orientation == 7:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(image, 0)
    elif orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 旋转270度
    else:
        return image


def process_image_base(image_data, process_func, process_name, **kwargs):
    """改进的基础图像处理函数，保持原始尺寸和方向"""
    try:
        start_time = time.time()

        # 获取原始图像方向
        orientation = get_image_orientation(image_data)

        # 解码图像
        img = Image.open(io.BytesIO(image_data))
        original_size = img.size  # 保存原始尺寸

        # 转换为OpenCV格式
        img_nd = np.array(img)

        # 确保图像为 BGR 格式（OpenCV 标准）
        if len(img_nd.shape) == 3:
            if img_nd.shape[2] == 4:  # RGBA
                img_nd = cv2.cvtColor(img_nd, cv2.COLOR_RGBA2BGR)
            elif img_nd.shape[2] == 3:  # RGB
                img_nd = cv2.cvtColor(img_nd, cv2.COLOR_RGB2BGR)

        # 应用方向校正
        img_nd = apply_orientation(img_nd, orientation)

        # 应用处理函数
        processed_image = process_func(img_nd, **kwargs)

        # 确保输出图像尺寸与输入一致
        if processed_image.shape[:2] != img_nd.shape[:2]:
            processed_image = cv2.resize(processed_image, (original_size[0], original_size[1]))

        # 转换为 base64
        result_base64 = image_to_base64(processed_image)

        return {
            'success': True,
            'processed_image': f"data:image/jpeg;base64,{result_base64}",
            'timestamp': time.time(),
        }

    except Exception as e:
        logger.error(f"{process_name} 处理失败: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        }
