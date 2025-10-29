# 文档扫描服务

这是一个基于Django和YOLOv11的文档扫描服务，用户可以上传图片，系统会使用预训练的YOLOv11分割模型进行文档检测和分割。

## 功能特性

- 支持图片上传和文档扫描
- 使用YOLOv11分割模型进行文档检测
- 提供JSON格式的检测结果
- 包含简单的Web界面用于测试

## 安装说明

1. 确保已安装Python 3.8或更高版本

2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```

3. 准备模型文件：
   - 将预训练的`yolo11s-seg.pt`模型文件放在项目根目录

## 使用方法

1. 启动开发服务器：
   ```
   python manage.py runserver
   ```

2. 访问Web界面：
   - 打开浏览器，访问 `http://localhost:8000/api/`
   - 上传图片并点击"扫描文档"按钮

3. API接口使用：
   - 扫描接口：`POST /api/scan/`
     - 请求体：JSON格式，包含base64编码的图片
     - 响应：JSON格式，包含检测结果
   - 健康检查接口：`GET /api/health/`
     - 响应：包含服务和模型状态信息

## API响应格式

### 扫描接口响应
```json
{
  "status": "success",
  "detections": [
    {
      "box": [x1, y1, x2, y2],  // 边界框坐标
      "confidence": 0.95,      // 置信度
      "class": 0,              // 类别ID
      "mask": [[x1,y1], [x2,y2], ...]  // 分割掩码坐标（如果有）
    }
  ]
}
```

### 健康检查接口响应
```json
{
  "status": "ok",
  "model_status": "模型已加载",
  "model_file_exists": true
}
```

## 注意事项

- 确保`yolo11s-seg.pt`模型文件存在于项目根目录
- 对于大文件，可能需要调整Django的上传限制
- 生产环境部署时，建议配置适当的安全措施