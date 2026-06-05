"""
Agent Demo 配置模块
从环境变量加载配置，.env 文件优先级最高。
"""
import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    api_key = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
    # DeepSeek 新版 base_url 不含 /v1，OpenAI SDK 会自动补全路径
    # 旧版 https://api.deepseek.com/v1 仍兼容，但官方推荐不带 /v1
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    # deepseek-chat 将于 2026/07/24 弃用，届时改为 deepseek-v4-pro
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    temperature = 0.1  # 低温度保证指令解析的稳定性
    max_tokens = 2048
    timeout = 60


class DroneConfig:
    device_id = os.getenv("DRONE_DEVICE_ID", "DJI-Matrice-001")
    battery = int(os.getenv("DRONE_BATTERY", "85"))
    home_lat = float(os.getenv("DRONE_HOME_LAT", "31.025"))
    home_lng = float(os.getenv("DRONE_HOME_LNG", "121.435"))


llm_config = LLMConfig()
drone_config = DroneConfig()
