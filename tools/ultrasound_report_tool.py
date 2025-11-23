# File: tools/ultrasound_report_tool_openai.py
from openai import OpenAI
from langchain.agents import Tool
import base64
from PIL import Image
import io
import os
import json

# 初始化 OpenAI 客户端连接 vLLM 服务
client = OpenAI(
    base_url="",  # 替换为你的服务器地址
    api_key="no-api-key-required"
)

def encode_image(image_path):
    """处理图像并转换为Base64编码，自动调整大小"""
    img = Image.open(image_path)

    # 转换为JPEG格式（减少体积）
    buffered = io.BytesIO()
    if image_path.lower().endswith('.png'):
        img_format = "PNG"
    else:
        img_format = "JPEG"  # 默认用JPEG
    img.save(buffered, format=img_format)

    return base64.b64encode(buffered.getvalue()).decode('utf-8'), img_format

def generate_ultrasound_report(image_paths: str) -> str:
    # image_paths = json.loads(image_paths)
    # 构建多图消息内容
    image_paths = [image_paths]
    image_content = []
    for img_path in image_paths[:4]:  # 确保不超过四张
        try:
            img_b64, img_format = encode_image(img_path)
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{img_format.lower()};base64,{img_b64}",
                    "detail": "auto"  # 可选: low/high/auto
                }
            })
            print(f"✓ 已加载图片: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"× 图片加载失败 [{img_path}]: {str(e)}")

    # 完整的消息内容（文本+多图）
    content = [
                  {"type": "text",
                   "text": "请根据以上几张同一患者的超声波成像图，给出一份专业的医学报告。可通过影像顶部文字判断左右眼。"}
              ] + image_content  # 将文本和多图内容合并

    # 创建消息
    messages = [{
        "role": "user",
        "content": content
    }]

    # 发送请求
    response = client.chat.completions.create(
        model="./models/qwen_export",
        messages=messages,
        max_tokens=20480,
        temperature=0.7
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


ultrasound_report_tool = Tool(
    name="Ultrasound Report Generator (vLLM)",
    func=generate_ultrasound_report,
    description=(
        "使用本地图像（只能眼科B超图像！）调用通过vLLM部署的Qwen多模态模型生成医学报告。"
        "只能传入单张图像的路径"
        "通过OpenAI兼容接口传递图像Base64格式。"
    )
)

if __name__ == '__main__':
    B_path = ""
    generate_ultrasound_report(B_path)

