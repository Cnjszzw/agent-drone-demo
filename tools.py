"""
无人机工具集 —— LangChain @tool 装饰器

核心设计原则（面试重点）:

1. Agent 工具暴露的是业务语义，不是硬件原语。
   - record_for_duration(60)   ← LLM 看到的（业务语义）
   - start → wait → stop       ← 底层实现的（硬件原语，LLM 不感知）

2. 能硬编码的规则不交给 LLM 决策。
   - SafetyGate 校验 → 硬编码
   - 飞行前通知前端画预览线 → 硬编码（内嵌在 fly_to_point 里）
   - Human-in-the-loop 确认 → 硬编码
   曾经把通知前端作为独立 Tool（notify_frontend），但 LLM 偶发漏调。
   根因: 飞行前通知是确定性规则，不应交给概率模型决策。

3. LLM 只负责意图理解，不负责安全决策和时序编排。

生产化路径:
- MockExecutor → MqttExecutor（方法签名零改动）
- _notify_handler → POST /api/wvp/agent/notify → Java WS → 前端
- SafetyGate 规则从配置文件加载（支持运行时更新）
"""
import json
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

# ==================== Human-in-the-loop 确认 ====================

_confirm_handler = lambda prompt: input(prompt).strip().lower() == "y"


def set_confirm_handler(handler):
    global _confirm_handler
    _confirm_handler = handler


def _confirm(prompt: str, risk: str = "high") -> bool:
    result = _confirm_handler(prompt)
    if result:
        logger.warning("⚠️ [%s风险] 操作已确认执行", risk)
    else:
        logger.info("❌ [%s风险] 用户取消操作", risk)
    return result


# ==================== 前端通知回调（模拟生产环境） ====================

_notify_handler = lambda event_type, data: logger.info(
    "📢 [通知前端] %s: %s", event_type, json.dumps(data, ensure_ascii=False)
)


def set_notify_handler(handler):
    """注入前端通知回调。生产环境调 Java wvp-server 的 WebSocket 接口。"""
    global _notify_handler
    _notify_handler = handler


def _notify(event_type: str, data: dict):
    """
    通知前端。走 Python → Java HTTP → Java WebSocket → 前端 中转链路。
    不复用 Python 直推前端的原因: 前端已有 Java WS 长连接（鉴权/心跳/Session），
    Python 再开一条 WS 需要重复实现鉴权，没必要。
    """
    logger.info("📢 [通知前端] %s", event_type)
    _notify_handler(event_type, data)


def _on_fly_progress(task_id: str, progress: int, status: str):
    """飞行进度回调 → 通知前端更新进度条"""
    _notify("flight_progress", {
        "task_id": task_id,
        "progress": progress,
        "status": status,
    })


# ==================== 工具定义 ====================


@tool
def fly_to_point(lat: float, lng: float, height: float) -> str:
    """
    控制无人机飞向指定的 GPS 坐标位置。飞行是异步过程（分钟级），
    工具内部封装轮询等待，LLM 只看到最终结果（到达/失败/超时）。

    对应 WVP: DrcController → MQTT topic: dji/device/{sn}/control/fly

    Args:
        lat: 目标纬度，范围 18-54（中国境内）
        lng: 目标经度，范围 73-135（中国境内）
        height: 飞行高度（米），相对地面，范围 10-120
    """
    logger.info("🛫 飞行指令: (%.6f, %.6f) 高度 %.0fm", lat, lng, height)

    # 1. SafetyGate 校验（硬编码，不经过 LLM）
    result = safety_gate.validate_fly(lat, lng, height)
    if not result.passed:
        return f"❌ 飞行指令被拒绝: {result.reason}"
    if result.warning:
        logger.warning("⚠️ %s", result.reason)

    # 2. 通知前端画预览线和目标点（硬编码 —— 曾经是独立 Tool，后因 LLM 漏调下沉至此）
    _notify("flight_preview", {
        "lat": lat, "lng": lng, "height": height,
        "from_lat": executor.current_lat,
        "from_lng": executor.current_lng,
    })
    logger.info("  📍 已通知前端展示飞行预览线")

    # 3. Human-in-the-loop 确认
    if not _confirm(
        f"  ⚠️ 确认飞至 ({lat:.6f}, {lng:.6f}) 高度 {height}m? (y/n): "
    ):
        return "❌ 用户取消飞行"

    # 4. 执行飞行（内部封装轮询等待 + 进度推送）
    return executor.fly_to_point(lat, lng, height,
                                 on_progress=_on_fly_progress)


@tool
def record_for_duration(duration_seconds: int) -> str:
    """
    录制指定时长的视频。底层自动处理 start → 等待 → stop 流程。

    为什么不是 start_recording + stop_recording 两个独立 Tool:
    LLM 没有时间感知能力，无法可靠编排"开始→等待60s→停止"的时序。
    工具暴露的是业务语义（录制X秒），不是硬件原语（开始/停止）。

    对应 WVP: CameraRecordingStartImpl + CameraRecordingStopImpl

    Args:
        duration_seconds: 录制时长（秒），范围 5-300
    """
    logger.info("🎥 录制指令: %ds", duration_seconds)
    return executor.record_for_duration(duration_seconds)


@tool
def start_recording() -> str:
    """
    开始持续录像（不指定时长，需手动停止）。
    用于需要手动控制录像时长的场景，如"一直录到我说停"。

    注意: 如果用户指定了明确的录像时长（如"录60秒"），
    应优先使用 record_for_duration 工具。

    对应 WVP: CameraRecordingStartImpl
    """
    logger.info("🎥 开始持续录像")
    return executor.start_recording()


@tool
def stop_recording() -> str:
    """
    停止正在进行的录像。配合 start_recording 使用。

    对应 WVP: CameraRecordingStopImpl
    """
    logger.info("⏹️ 停止录像")
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
def set_zoom(factor: float) -> str:
    """
    调节相机变焦倍数。可用于放大观察远处目标或缩小获取更广视野。

    对应 WVP: CameraFocalLengthSetImpl / CameraFrameZoomImpl

    Args:
        factor: 变焦倍数，范围 1-56（红外镜头固定 1x）
    """
    logger.info("🔍 变焦指令: %.1fx", factor)

    if factor < 1 or factor > 56:
        return f"❌ 变焦倍数 {factor}x 超出范围，支持 1x-56x"

    if executor.lens_mode == "ir" and factor != 1:
        return "❌ 红外镜头不支持变焦，请先切换到广角或变焦镜头"

    return executor.set_zoom(factor)


@tool
def switch_lens(mode: str) -> str:
    """
    切换相机镜头类型。支持广角（大场景）、变焦（远距离观察）、红外（热成像）三种。

    对应 WVP: CameraModeSwitchImpl

    Args:
        mode: 镜头类型，'wide'=广角, 'zoom'=变焦, 'ir'=红外（热成像）
    """
    logger.info("📷 镜头切换: %s", mode)

    if mode not in ("wide", "zoom", "ir"):
        return f"❌ 不支持的镜头类型: {mode}，仅支持 wide/zoom/ir"

    if mode == "ir" and executor.recording:
        return "❌ 录像中无法切换镜头，请先停止录像"

    return executor.switch_lens(mode)


@tool
def panorama_photo() -> str:
    """
    全景拍照。无人机自动拍摄多张照片并合成为全景图。
    拍摄期间无人机需保持稳定悬停（实际约 60s，Demo 加速为 4s）。

    ⚠️ 全景拍摄期间请勿移动无人机，否则合成失败。

    对应 WVP: CameraPhotoTakeImpl (panorama mode)
    """
    logger.info("🖼️ 全景拍照指令")
    return executor.panorama_photo()


@tool
def set_exposure_mode(mode: str) -> str:
    """
    切换相机曝光模式。

    对应 WVP: CameraModeSwitchImpl (exposure sub-mode)

    Args:
        mode: 'auto'=自动曝光（可调节 ISO/快门/曝光补偿）
              'manual'=手动曝光（锁定参数，不可调节）
    """
    logger.info("☀️ 曝光模式: %s", mode)

    if mode not in ("auto", "manual"):
        return "❌ 不支持的曝光模式: 仅支持 auto（自动）和 manual（手动）"

    return executor.set_exposure_mode(mode)


@tool
def set_iso(value: int) -> str:
    """
    设置相机 ISO 感光度。仅在自动曝光模式下可用。

    对应 WVP: 相机参数控制

    Args:
        value: ISO 值，范围 100-25600
    """
    logger.info("📸 ISO: %d", value)

    if executor.exposure_mode == "manual":
        return ("❌ 当前为手动曝光模式，无法调节 ISO。"
                "请先调用 set_exposure_mode('auto') 切换到自动曝光模式")

    if value < 100 or value > 25600:
        return f"❌ ISO {value} 超出范围，支持 100-25600"

    return executor.set_iso(value)


@tool
def set_shutter_speed(speed: str) -> str:
    """
    设置相机快门速度。仅在自动曝光模式下可用。

    对应 WVP: 相机参数控制

    Args:
        speed: 快门速度，如 '1/60' '1/100' '1/500' '1/1000'
    """
    logger.info("📸 快门: %s", speed)

    if executor.exposure_mode == "manual":
        return ("❌ 当前为手动曝光模式，无法调节快门。"
                "请先调用 set_exposure_mode('auto') 切换到自动曝光模式")

    return executor.set_shutter_speed(speed)


@tool
def set_ev_compensation(ev: float) -> str:
    """
    设置曝光补偿，用于调整画面亮度。仅在自动曝光模式下可用。
    正值变亮，负值变暗。

    对应 WVP: 相机参数控制

    Args:
        ev: 曝光补偿值，范围 -3.0 ~ +3.0
    """
    logger.info("📸 曝光补偿: %+.1fEV", ev)

    if executor.exposure_mode == "manual":
        return ("❌ 当前为手动曝光模式，无法调节曝光补偿。"
                "请先调用 set_exposure_mode('auto') 切换到自动曝光模式")

    if ev < -3.0 or ev > 3.0:
        return f"❌ 曝光补偿 {ev}EV 超出范围，支持 -3.0 ~ +3.0"

    return executor.set_ev_compensation(ev)


@tool
def get_camera_status() -> str:
    """
    查询相机当前参数，包括镜头、变焦、曝光模式、ISO、快门、
    曝光补偿、录像状态、存储空间等。

    对应 WVP: manage 模块相机遥测数据
    """
    logger.info("📷 查询相机状态: %s", drone_config.device_id)
    return executor.get_camera_status()


@tool
def get_drone_status() -> str:
    """
    查询无人机当前状态，包括位置、电量、飞行模式、GNSS 信号、
    飞行任务状态、录像状态等。

    对应 WVP: manage 模块 OSD 遥测数据（Redis 缓存 + MQTT 实时推送）
    """
    logger.info("📡 查询设备: %s", drone_config.device_id)
    return executor.get_status()


# ==================== 工具注册 ====================

ALL_TOOLS = [
    fly_to_point,
    record_for_duration,
    start_recording,
    stop_recording,
    take_photo,
    panorama_photo,
    set_zoom,
    switch_lens,
    set_exposure_mode,
    set_iso,
    set_shutter_speed,
    set_ev_compensation,
    return_home,
    gimbal_control,
    get_drone_status,
    get_camera_status,
]
