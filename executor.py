"""
模拟执行器 —— 模拟 MQTT 下发到无人机硬件的完整效果

接口设计对应 WVP 真实控制链路:
- 每个方法构建 MQTT topic + JSON payload
- 打印 MQTT 发布日志（生产环境 publish 到 EMQX broker）
- 模拟硬件响应延迟
- 返回结构化的执行结果

关键设计（面试重点）:
- fly_to_point: 内部封装轮询等待（生产环境轮询 Redis 任务状态）
- record_for_duration: 复合工具，封装 start → wait → stop 流程
  不把原子指令暴露给 LLM 做时序编排（LLM 无时间感知能力）

生产化路径:
- MockExecutor 替换为 MqttExecutor，方法签名零改动
- 轮询目标从 Mock 状态改为 Redis/后端接口
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

        # 模拟当前飞行任务
        self._fly_task_id = None
        self._fly_task_status = None

        # 模拟当前位置
        self.current_lat = home_lat
        self.current_lng = home_lng
        self.current_height = 0.0

    # ==================== 飞行控制（异步轮询模式） ====================

    def fly_to_point(self, lat: float, lng: float, height: float,
                     on_progress=None) -> str:
        """
        指点飞行 —— 内部封装异步轮询等待。

        生产环境实际流程:
          1. POST /api/drone/fly → 返回 task_id
          2. 每秒轮询 Redis 查询任务状态
          3. 到达/失败后返回结果给 LLM
          4. 同时推送进度给前端（通过现有 WS 通道）

        LLM 只看到最终结果，不感知内部的轮询等待过程。
        """
        task_id = self._msg_id()
        topic = f"dji/device/{self.device_id}/control/fly"
        payload = (
            f'{{"msgId":"{task_id}",'
            f'"lat":{lat:.6f},"lng":{lng:.6f},"height":{height:.1f}}}'
        )
        self._print_mqtt(topic, payload)

        self._fly_task_id = task_id
        self._fly_task_status = "in_progress"

        # 模拟异步飞行 + 轮询（生产环境用 Redis/后端接口）
        total_steps = 5 + random.randint(0, 3)  # 5-8 秒模拟
        for step in range(1, total_steps + 1):
            time.sleep(1)
            progress = step * 100 // total_steps

            # 进度回调 → 通知前端（复现生产环境中通过 WS 推送进度）
            if on_progress:
                on_progress(task_id, progress, "in_progress")

            bar = "▮" * step + "▯" * (total_steps - step)
            print(f"\r  ⏳ 飞行中 [{bar}] {progress}%  ETA: {total_steps - step}s",
                  end="", flush=True)

            # 模拟偶发故障（5% 概率，除第一步外）
            if step > 1 and random.random() < 0.05:
                print()
                self._fly_task_status = "failed"
                return (f"❌ 飞行异常 [task: {task_id}]: GPS 信号丢失，"
                        f"无人机已触发自动悬停，请检查状态后重试")

        print()
        self._fly_task_status = "completed"

        # 更新模拟位置
        self.current_lat = lat
        self.current_lng = lng
        self.current_height = height
        self.battery -= 5 + random.randint(0, 10)

        return (f"✅ [task: {task_id}] 已到达目标点 ({lat:.6f}, {lng:.6f})，"
                f"当前高度 {height:.1f}m，电量 {max(0, self.battery)}%")

    def get_fly_task_status(self) -> dict:
        """查询飞行任务状态（模拟生产环境轮询接口）"""
        return {
            "task_id": self._fly_task_id,
            "status": self._fly_task_status or "idle",
        }

    # ==================== 录像控制（复合工具） ====================

    def record_for_duration(self, duration: int) -> str:
        """
        录制指定时长的视频 —— 复合工具。

        生产环境实际流程:
          底层硬件只有 start 和 stop 两个原子指令。
          此工具封装: start → 等待 duration 秒 → stop
          LLM 不需要知道底层有两个指令，也不负责编排时序。

        面试原则: Agent 工具暴露的是业务语义（record_for_duration），
        不是硬件原语（start_recording / stop_recording）。
        """
        # 1. 开始录像
        topic_start = f"dji/device/{self.device_id}/camera/record/start"
        payload_start = f'{{"msgId":"{self._msg_id()}"}}'
        self._print_mqtt(topic_start, payload_start)
        self.recording = True

        # 2. 等待指定时长（Demo 加速：duration/10，最少 1s，最多 8s）
        sim_sec = max(1, min(duration // 10, 8))
        for step in range(1, sim_sec + 1):
            time.sleep(1)
            bar = "▮" * step + "▯" * (sim_sec - step)
            print(f"\r  ⏳ 录像中 [{bar}] {step}/{sim_sec}s (实际录制 {duration}s)",
                  end="", flush=True)

            # 模拟异常中断（3% 概率）
            if random.random() < 0.03:
                print()
                self.recording = False
                return "❌ 录像异常中断: 存储卡已满，请清理后重试"

        print()

        # 3. 停止录像
        topic_stop = f"dji/device/{self.device_id}/camera/record/stop"
        payload_stop = f'{{"msgId":"{self._msg_id()}"}}'
        self._print_mqtt(topic_stop, payload_stop)
        self.recording = False

        return (f"✅ 录像完成，时长 {duration}s，"
                f"文件: REC_{int(time.time()) % 100000}.mp4")

    def start_recording(self) -> str:
        """
        开始持续录像（不指定时长，手动停止）。
        保留此方法给需要手动控制录像时长的场景。
        """
        topic = f"dji/device/{self.device_id}/camera/record/start"
        payload = f'{{"msgId":"{self._msg_id()}"}}'
        self._print_mqtt(topic, payload)
        self.recording = True
        return f"✅ 开始持续录像，设备: {self.device_id}（使用 stop_recording 停止）"

    def stop_recording(self) -> str:
        """停止录像"""
        topic = f"dji/device/{self.device_id}/camera/record/stop"
        payload = f'{{"msgId":"{self._msg_id()}"}}'
        self._print_mqtt(topic, payload)
        if self.recording:
            self.recording = False
            return f"✅ 录像已停止，设备: {self.device_id}"
        return "ℹ️ 当前没有正在进行的录像"

    # ==================== 拍照 ====================

    def take_photo(self, count: int) -> str:
        topic = f"dji/device/{self.device_id}/camera/photo"
        for i in range(1, count + 1):
            payload = (f'{{"msgId":"{self._msg_id()}",'
                       f'"index":{i},"total":{count}}}')
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

        total = 3 + random.randint(0, 2)
        for step in range(1, total + 1):
            time.sleep(1)
            bar = "▮" * step + "▯" * (total - step)
            print(f"\r  ⏳ 返航中 [{bar}] {step}/{total}s", end="", flush=True)
        print()

        self.current_lat = self.home_lat
        self.current_lng = self.home_lng
        self.current_height = 0.0
        self.battery -= 3 + random.randint(0, 5)

        return (f"✅ 已返航降落至起飞点 ({self.home_lat:.6f}, "
                f"{self.home_lng:.6f})，电量 {max(0, self.battery)}%")

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
        fly_status = (f"飞行任务: {self._fly_task_status}"
                      if self._fly_task_id else "无进行中的飞行任务")
        return (
            f"📡 设备状态 [{self.device_id}]:\n"
            f"  位置: ({self.current_lat:.6f}, {self.current_lng:.6f})\n"
            f"  高度: {self.current_height:.1f}m（相对地面）\n"
            f"  电量: {max(0, self.battery)}%\n"
            f"  GNSS: 良好 (28颗星)\n"
            f"  飞行模式: {mode}\n"
            f"  录像状态: {rec}\n"
            f"  {fly_status}\n"
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
                print(f"\r  ⏳ {action} [{bar}] {i}/{seconds}s",
                      end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\n  ⚠️ 操作被中断")
            raise
