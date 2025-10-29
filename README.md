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

## 使用方法

1. 启动开发服务器：
   ```
   python manage.py runserver
   ```

2. 访问Web界面：
    - 打开浏览器，访问 `http://localhost:8000`
    - 上传图片并点击"扫描文档"按钮

## 功能测试页面

访问 [`http://localhost:8000/`](http://localhost:8000/) 即可

## API响应格式

### 文档识别

#### 请求

接口：[`POST /api/scan`](#文档识别)

请求体：form-data格式

```json
{
  "image": "Image File"
}
```

#### 响应

```json
{
  "data": "Image Base64",
  "message": "文档扫描完成",
  "status": "success"
}
```

### 灰度处理

#### 请求

接口：[`POST /api/grayscale`](#灰度处理)

```json
{
  
}
```

#### 响应

```json
{
}
```
### 增强处理

#### 请求

接口：[`POST /api/enhance`](#增强处理)

```json
{
  
}
```

#### 响应

```json
{
  
}
```
### 锐化处理

#### 请求

接口：[`POST /api/sharpen`](#锐化处理)

```json
{
  
}
```

#### 响应

```json
{
  
}
```
### 黑白处理

#### 请求

接口：[`POST /api/black-white`](#黑白处理)

```json
{
  
}
```

#### 响应

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

- [x] 图片转灰度
- [x] 图片锐化
- [x] 图片增强
- [x] 图片黑白

> 👆👆👆👆👆需要优化👆👆👆👆👆