"""
模拟执行器 —— 模拟 MQTT 下发到无人机硬件的完整效果

接口设计：
- 每个方法对应一个 MQTT topic，构建 topic + JSON payload
- 打印 MQTT 发布日志（真实场景 publish 到 EMQX broker）
- 模拟硬件响应延迟
- 返回结构化的执行结果

生产化路径：
- 替换为 MqttExecutor，实现同一套方法签名
- 调用 Java wvp-server 的 REST API（POST /api/drone/control/xxx）
- 或直接通过 paho-mqtt 发到 EMQX broker

面试时：
"Demo 阶段用 MockExecutor 验证工具链路正确性。
验证通过后只需实现 MqttExecutor 替换，工具函数签名零改动。"
"""
import time
import random
import uuid


class MockExecutor:
    def __init__(self, device_id: str, home_lat: float, home_lng: float, battery: int):
        self.device_id = device_id
        self.home_lat = home_lat
        self.home_lng = home_lng
        self.battery = battery
        self.recording = False

        # 模拟当前位置
        self.current_lat = home_lat
        self.current_lng = home_lng
        self.current_height = 0.0

    # ==================== 飞行控制 ====================

    def fly_to_point(self, lat: float, lng: float, height: float) -> str:
        topic = f"dji/device/{self.device_id}/control/fly"
        payload = (
            f'{{"msgId":"{self._msg_id()}",'
            f'"lat":{lat:.6f},"lng":{lng:.6f},"height":{height:.1f}}}'
        )
        self._print_mqtt(topic, payload)

        flight_time = 2 + random.randint(0, 3)
        self._progress("飞行中", flight_time)

        self.current_lat = lat
        self.current_lng = lng
        self.current_height = height
        self.battery -= 5 + random.randint(0, 10)

        return f"✅ 已到达目标点 ({lat:.6f}, {lng:.6f})，当前高度 {height:.1f}m，电量 {max(0, self.battery)}%"

    # ==================== 相机控制 ====================

    def start_recording(self, duration: int) -> str:
        topic = f"dji/device/{self.device_id}/camera/record/start"
        payload = f'{{"msgId":"{self._msg_id()}","duration":{duration}}}'
        self._print_mqtt(topic, payload)
        self.recording = True

        if duration > 0:
            sim_sec = min(duration // 10, 6)  # Demo 加速
            self._progress("录像中", sim_sec)
            self.recording = False
            return f"✅ 录像完成，时长 {duration}s，文件: REC_{int(time.time()) % 100000}.mp4"
        return f"✅ 开始持续录像，设备: {self.device_id}"

    def stop_recording(self) -> str:
        topic = f"dji/device/{self.device_id}/camera/record/stop"
        payload = f'{{"msgId":"{self._msg_id()}"}}'
        self._print_mqtt(topic, payload)
        if self.recording:
            self.recording = False
            return f"✅ 录像已停止，设备: {self.device_id}"
        return "ℹ️ 当前没有正在进行的录像"

    def take_photo(self, count: int) -> str:
        topic = f"dji/device/{self.device_id}/camera/photo"
        for i in range(1, count + 1):
            payload = f'{{"msgId":"{self._msg_id()}","index":{i},"total":{count}}}'
            self._print_mqtt(topic, payload)
            self._progress(f"拍照 {i}/{count}", 1)

        if count == 1:
            return f"✅ 拍照完成，照片: PHOTO_{int(time.time()) % 100000}.jpg"
        return f"✅ 连拍完成，共 {count} 张照片"

    # ==================== 返航 ====================

    def return_home(self) -> str:
        topic = f"dji/device/{self.device_id}/control/return_home"
        payload = (
            f'{{"msgId":"{self._msg_id()}",'
            f'"homeLat":{self.home_lat:.6f},"homeLng":{self.home_lng:.6f}}}'
        )
        self._print_mqtt(topic, payload)
        self._progress("返航中", 3)

        self.current_lat = self.home_lat
        self.current_lng = self.home_lng
        self.current_height = 0.0
        self.battery -= 3 + random.randint(0, 5)

        return f"✅ 已返航降落至起飞点 ({self.home_lat:.6f}, {self.home_lng:.6f})，电量 {max(0, self.battery)}%"

    # ==================== 云台控制 ====================

    def gimbal_control(self, mode: str) -> str:
        topic = f"dji/device/{self.device_id}/gimbal/control"
        angle = "0" if mode == "center" else "-90"
        desc = "回中（水平）" if mode == "center" else "垂直向下（-90°）"
        payload = f'{{"msgId":"{self._msg_id()}","pitch":{angle},"yaw":0}}'
        self._print_mqtt(topic, payload)
        self._progress("云台调整", 1)
        return f"✅ 云台已调整为 {desc}"

    # ==================== 状态查询 ====================

    def get_status(self) -> str:
        mode = "空中悬停" if self.current_height > 0 else "地面待命"
        rec = "录像中" if self.recording else "未录像"
        return (
            f"📡 设备状态 [{self.device_id}]:\n"
            f"  位置: ({self.current_lat:.6f}, {self.current_lng:.6f})\n"
            f"  高度: {self.current_height:.1f}m（相对地面）\n"
            f"  电量: {max(0, self.battery)}%\n"
            f"  GNSS: 良好 (28颗星)\n"
            f"  飞行模式: {mode}\n"
            f"  录像状态: {rec}\n"
            f"  信号强度: -42dBm (优秀)"
        )

    # ==================== 内部工具 ====================

    @staticmethod
    def _msg_id() -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _print_mqtt(topic: str, payload: str):
        print()
        print("┌─────────────────────────────────────────────")
        print(f"│ 📤 [MQTT Publish]")
        print(f"│ Topic  : {topic}")
        print(f"│ Payload: {payload}")
        print("└─────────────────────────────────────────────")

    @staticmethod
    def _progress(action: str, seconds: int):
        try:
            for i in range(1, seconds + 1):
                time.sleep(1)
                bar = "▮" * i + "▯" * (seconds - i)
                print(f"\r  ⏳ {action} [{bar}] {i}/{seconds}s", end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\n  ⚠️ 操作被中断")
            raise
