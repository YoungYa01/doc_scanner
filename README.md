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

## 功能测试页面

访问 [`http://localhost:8000/`](http://localhost:8000/) 即可

## API响应格式

### 扫描接口响应

等待定义...

```json
{
  
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

## todo list

- [ ] 图片转灰度
- [ ] 图片锐化
- [ ] 图片增强
- [ ] 图片黑白