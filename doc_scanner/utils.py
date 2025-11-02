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
    """图像增强处理 - 文档扫描增强算法"""
    try:
        result = image.copy()
    
        # 1. 黑白文字处理（二值化）
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # # 自适应二值化（高斯加权法，更抗噪）
        # binary = cv2.adaptiveThreshold(
        #     gray, 
        #     255, 
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权局部阈值
        #     cv2.THRESH_BINARY, 
        #     blockSize=21,  # 邻域窗口大小（需为奇数，建议5~21，文字细则选小，如7）
        #     C=5            # 常数偏移量（负数降低阈值，保留更多文字；正数提高阈值，过滤噪声）
        # )
        # result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # 1. 计算图像统计特征（亮度均值和标准差）
        mean_val = cv2.mean(gray)[0]  # 全局亮度均值（0~255）
        std_val = gray.std()          # 亮度标准差（值小说明对比度低）

        print(f"mean_val: {mean_val}, std_val: {std_val}")


        # 2. 动态调整blockSize（邻域窗口大小）
        if std_val < 40:  # 对比度低（噪声敏感），用更大窗口平滑噪声
            block_size = 21   # 窗口大→局部阈值更稳健，减少噪点
        else:  # 对比度高，用小窗口保留细节
            block_size = 11   # 窗口小→更灵敏捕捉局部细节

        # 确保blockSize为奇数
        block_size = block_size if block_size % 2 == 1 else block_size + 1

        # 3. 动态调整C（阈值偏移量）
        if mean_val < 100:  # 整体偏暗，降低阈值（负C）保留更多内容
            C = 9
        elif mean_val > 170:  # 整体偏亮，提高阈值（正C）过滤高光噪声
            C = 19
        else:  # 亮度适中，默认C
            C = 13

        # 4. 应用动态参数的自适应二值化
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权更抗噪
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 2. 变细处理（形态学腐蚀，参数0.1）
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.erode(result, kernel, iterations=1)
        
        # 3. 高斯模糊（半径1.2）
        # kernel_size = int(6 * 1.2 + 1)
        # if kernel_size % 2 == 0:
        #     kernel_size += 1
        # result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 1.2)
        blur_radius = 0.5  # 降低模糊强度
        kernel_size = int(6 * blur_radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 核尺寸变为5×5（6×0.8+1=5.8→6→7？不，0.8×6=4.8+1=5.8→取5，因5是奇数）
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), blur_radius)

        # 4. 高斯锐化（半径3, 阶数7）
        # 先模糊再锐化
        blur = cv2.GaussianBlur(result, (7, 7), 3)
        result = cv2.addWeighted(result, 2.0, blur, -1.0, 0)  # 差值更明显，边缘更锐
        
        # 5. 多尺度细节增强（18）
        # 使用多个尺度的拉普拉斯算子增强细节
        for scale in range(1, 4):
            # 不同尺度的增强
            laplacian = cv2.Laplacian(result, cv2.CV_64F, ksize=5)
            laplacian_abs = np.uint8(np.absolute(laplacian))
            # 根据参数18调整增强强度
            result = cv2.addWeighted(result, 1.0, laplacian_abs, 0.12, 0)  # 强度从0.18提升到0.3
        
        # 6. USM锐化
        result = process_sharpen(result)
        return result
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
