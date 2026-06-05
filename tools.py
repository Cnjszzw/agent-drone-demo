"""
无人机工具集 —— LangChain @tool 装饰器

每个 @tool 函数自动生成 OpenAI Function Calling Schema，由 AgentExecutor
注入到 LLM 的 tools 参数中。LLM 返回 tool_calls 时，框架自动调用对应函数。

对应 WVP 真实功能的映射见每个函数的 docstring。

Human-in-the-loop 确认机制：
- CLI 模式 (main.py)：确认回调 = input()，用户在终端输入 y/n
- API 模式 (app.py)：确认回调 = 自动确认 + 日志警告
  生产环境中 API 模式应改为：返回 pending 状态 → 前端弹出确认卡片 → 回调确认接口

生产化路径：
- 工具函数签名不变，只替换 executor 实现类（MockExecutor → MqttExecutor）
- safety_gate 替换为完整规则引擎版本
"""
import logging
from langchain_core.tools import tool
from config import drone_config
from safety import SafetyGate
from executor import MockExecutor

logger = logging.getLogger(__name__)

safety_gate = SafetyGate(drone_config.home_lat, drone_config.home_lng)
executor = MockExecutor(
    drone_config.device_id,
    drone_config.home_lat,
    drone_config.home_lng,
    drone_config.battery
)

# ==================== Human-in-the-loop 确认机制 ====================

# 确认回调：签名为 (prompt: str) -> bool
# CLI 模式设为 lambda p: input(p).strip().lower() == 'y'
# API 模式设为 lambda p: True（自动确认 + 日志警告）
_confirm_handler = lambda prompt: input(prompt).strip().lower() == "y"


def set_confirm_handler(handler):
    """注入确认回调。CLI 模式用 input()，API 模式用自动确认。"""
    global _confirm_handler
    _confirm_handler = handler


def _confirm(prompt: str, risk: str = "high") -> bool:
    """
    高风险操作确认。
    API 模式下自动确认并记录警告日志——生产环境应改为独立的确认流程：
    1. Agent 返回 pending_confirm 状态
    2. 前端弹出确认卡片
    3. 用户确认后回调 POST /api/agent/confirm
    """
    result = _confirm_handler(prompt)
    if result:
        logger.warning("⚠️ [%s风险] 操作已确认执行", risk)
    else:
        logger.info("❌ [%s风险] 用户取消操作", risk)
    return result


# ==================== 工具定义 ====================


@tool
def fly_to_point(lat: float, lng: float, height: float) -> str:
    """
    控制无人机飞向指定的 GPS 坐标位置。

    对应 WVP: DrcController → MQTT topic: dji/device/{sn}/control/fly
    高风险操作，执行前经过 SafetyGate 校验和人工确认。

    Args:
        lat: 目标纬度，范围 18-54（中国境内）
        lng: 目标经度，范围 73-135（中国境内）
        height: 飞行高度（米），相对地面，范围 10-120
    """
    logger.info("🛫 飞行指令: (%.6f, %.6f) 高度 %.0fm", lat, lng, height)

    result = safety_gate.validate_fly(lat, lng, height)
    if not result.passed:
        return f"❌ 飞行指令被拒绝: {result.reason}"
    if result.warning:
        logger.warning("⚠️ %s", result.reason)

    if not _confirm(f"  ⚠️ 确认飞至 ({lat:.6f}, {lng:.6f}) 高度 {height}m? (y/n): "):
        return "❌ 用户取消飞行"

    return executor.fly_to_point(lat, lng, height)


@tool
def start_recording(duration_seconds: int) -> str:
    """
    开始录像，可指定时长。录像期间无人机保持当前状态。

    对应 WVP: CameraRecordingStartImpl → MQTT topic: dji/device/{sn}/camera/record/start

    Args:
        duration_seconds: 录像时长（秒），0 表示持续录像直到手动停止
    """
    logger.info("🎥 录像指令: %ds", duration_seconds)
    return executor.start_recording(duration_seconds)


@tool
def stop_recording() -> str:
    """
    停止当前正在进行的录像。

    对应 WVP: CameraRecordingStopImpl → MQTT topic: dji/device/{sn}/camera/record/stop
    """
    logger.info("⏹️ 停止录像指令")
    return executor.stop_recording()


@tool
def take_photo(count: int) -> str:
    """
    拍照，可指定连拍张数。

    对应 WVP: CameraPhotoTakeImpl → MQTT topic: dji/device/{sn}/camera/photo

    Args:
        count: 拍照张数，默认 1 张
    """
    logger.info("📸 拍照指令: %d张", count)
    return executor.take_photo(count)


@tool
def return_home() -> str:
    """
    无人机自动返回起飞点并降落。

    对应 WVP: DockController → MQTT topic: dji/device/{sn}/control/return_home
    高风险操作，必须是任务最后一步，返航后不应再执行其他指令。
    """
    logger.info("🏠 返航指令")

    result = safety_gate.validate_return_home()
    if not result.passed:
        return f"❌ 返航被拒绝: {result.reason}"

    if not _confirm("  ⚠️ 确认返航降落? (y/n): "):
        return "❌ 用户取消返航"

    return executor.return_home()


@tool
def gimbal_control(mode: str) -> str:
    """
    控制无人机云台角度。

    对应 WVP: GimbalResetImpl → MQTT topic: dji/device/{sn}/gimbal/control

    Args:
        mode: 'center' = 回中（水平），'down' = 垂直向下（90度）
    """
    logger.info("🎯 云台指令: %s", mode)

    if mode not in ("center", "down"):
        return f"❌ 不支持的云台模式: {mode}，仅支持 center（回中）和 down（垂直向下）"

    return executor.gimbal_control(mode)


@tool
def get_drone_status() -> str:
    """
    查询无人机当前状态，包括位置、电量、飞行模式、GNSS 信号等。

    对应 WVP: manage 模块 OSD 遥测数据查询（Redis 缓存 + MQTT 实时推送）
    """
    logger.info("📡 查询设备: %s", drone_config.device_id)
    return executor.get_status()


# 注册所有工具
ALL_TOOLS = [
    fly_to_point,
    start_recording,
    stop_recording,
    take_photo,
    return_home,
    gimbal_control,
    get_drone_status,
]
