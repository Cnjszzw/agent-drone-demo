"""
模拟执行器 —— 模拟 MQTT 下发 + Redis 轮询状态

真实生产链路（本 Demo 模拟的对象）:

  嵌入式 → MQTT {status: "in_progress", progress: 60%}
      → Java wvp-server 消费 → 写入 Redis: drone:task:T001:status
      → Python Agent: redis-py GET drone:task:T001:status（<1ms, localhost）
      → 进度只推前端展示，LLM 不感知中间状态
      → 直到 status=completed/failed 才返回给 LLM

关键设计（面试重点）:
  1. Python 直连 Redis（redis-py），不绕 Java HTTP。
     原因: localhost 下 <1ms，比 HTTP 往返少一次序列化。
     Python 只是"观察者"（只读），不写入，不破坏数据一致性。
  2. 进度推前端，不推 LLM。
     LLM 不需要知道"飞了 60%"，它只关心终点：到达/失败/超时。
  3. fly_to_point 工具函数内部封装轮询等待。
     对 LLM 来说就是一次函数调用，返回了 = 完成了。

方案对比（面试用）:
  方案一: Python redis-py 直连 Redis 读状态 ← 当前选择
  方案二: Python paho-mqtt 直连 EMQX       ← 不选：需重复实现消息解析逻辑
  方案三: Python ↔ Java WS 长连接          ← 不选：连接管理复杂度高，收益被轮询抵消
  方案四: Java HTTP 回调 Python            ← 不选：飞行状态变化多次，回调只能触发一次
  最终选方案一。原因: 无新依赖、无连接管理、1s 轮询对 2-3 分钟飞行可忽略。

生产化路径:
  MockExecutor → 删除 mock_task_store → 替换为 redis-py GET
"""
import time
import random
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _MockTaskState:
    """模拟 Redis 中的任务状态。生产环境 key: drone:task:{task_id}:status"""
    task_id: str
    status: str = "started"       # started → in_progress → completed / failed
    progress: int = 0             # 0-100
    error: str = ""
    eta_seconds: int = 0          # 预计剩余秒数
    updated_at: float = 0.0


class MockExecutor:
    """
    模拟执行器。

    生产环境中的 Redis 轮询模式:
      task_state = redis.get(f"drone:task:{task_id}:status")
      # → b'{"status":"in_progress","progress":60,"eta_seconds":50}'
    """

    # 紧急停止标志（类级别，所有实例共享）
    # 生产环境: redis.set("agent:session:emergency_stop", "1")
    # 急停链路: 前端按钮 → Java /api/drone/emergency_stop → MQTT 直达无人机
    #          前端同时关闭 SSE 连接，Agent 会话自然终止
    _emergency_stop = False

    @classmethod
    def trigger_emergency_stop(cls):
        """触发紧急停止"""
        cls._emergency_stop = True

    @classmethod
    def reset_emergency_stop(cls):
        """重置停止标志（新会话开始时）"""
        cls._emergency_stop = False

    @classmethod
    def is_emergency_stopped(cls) -> bool:
        return cls._emergency_stop

    def __init__(self, device_id: str, home_lat: float, home_lng: float, battery: int):
        self.device_id = device_id
        self.home_lat = home_lat
        self.home_lng = home_lng
        self.battery = battery
        self.recording = False

        # 模拟 Redis —— 存放进行中的任务状态
        # 生产环境: redis.set(f"drone:task:{task_id}:status", json)
        self._mock_redis: dict[str, _MockTaskState] = {}

        # 模拟当前位置
        self.current_lat = home_lat
        self.current_lng = home_lng
        self.current_height = 0.0

        # 相机状态
        self.zoom_level = 1.0       # 变焦倍数（1x-56x）
        self.lens_mode = "zoom"     # wide / zoom / ir
        self.exposure_mode = "auto" # auto / manual（只有 auto 下可调参数）
        self.iso = 100              # ISO 100-25600
        self.shutter_speed = "1/60" # 快门速度
        self.ev_compensation = 0.0  # 曝光补偿 -3.0 ~ +3.0

    # ==================== Redis 模拟接口 ====================

    def _mock_redis_get(self, task_id: str) -> Optional[_MockTaskState]:
        """
        模拟: redis.get(f"drone:task:{task_id}:status")
        生产环境: redis-py → GET drone:task:T001:status → <1ms
        """
        return self._mock_redis.get(task_id)

    def _mock_redis_set(self, task_id: str, state: _MockTaskState):
        """模拟: redis.set() —— 实际由 Java 侧写入，Python 只读不写"""
        state.updated_at = time.time()
        self._mock_redis[task_id] = state

    def _start_progress_simulation(self, task_id: str, total_seconds: int):
        """
        启动后台线程模拟嵌入式硬件 → MQTT → Java → Redis 的异步进度更新。
        生产环境中这是真实硬件行为，Agent 不参与这个过程。
        """
        def simulate():
            state = self._mock_redis_get(task_id)
            if not state:
                return

            state.status = "started"
            self._mock_redis_set(task_id, state)

            for elapsed in range(1, total_seconds + 1):
                time.sleep(1)
                state = self._mock_redis_get(task_id)
                if not state:
                    return

                progress = min(elapsed * 100 // total_seconds, 99)
                state.status = "in_progress"
                state.progress = progress
                state.eta_seconds = total_seconds - elapsed
                self._mock_redis_set(task_id, state)

                # 模拟偶发故障（1% 概率，Demo 演示用；生产环境删除此逻辑）
                if elapsed > total_seconds * 0.3 and random.random() < 0.01:
                    state.status = "failed"
                    state.error = "GPS 信号丢失，无人机已触发自动悬停"
                    self._mock_redis_set(task_id, state)
                    return

            # 正常完成
            state = self._mock_redis_get(task_id)
            if state:
                state.status = "completed"
                state.progress = 100
                state.eta_seconds = 0
                self._mock_redis_set(task_id, state)

        t = threading.Thread(target=simulate, daemon=True)
        t.start()

    # ==================== 飞行控制 ====================

    def fly_to_point(self, lat: float, lng: float, height: float,
                     on_progress=None) -> str:
        """
        指点飞行 —— 内部封装 Redis 轮询等待 + 动态超时 + OSD 兜底。

        生产环境实际流程:
          1. 计算预估飞行时间（距离 / 平均速度），超时 = 预估 × 1.5
          2. POST /api/drone/control/fly → Java 下发 MQTT → 返回 task_id
          3. 每秒 redis.get(f"drone:task:{task_id}:status") 轮询
          4. 进度仅推前端，LLM 不感知中间状态
          5. 如果在预估时间内未完成 → 不直接报失败
             → 查 Redis OSD 实际位置: 是否距目标点 < 5m?
               是 → 判定到达（通信延迟导致的状态更新滞后）
               否 → 位置有变化(在飞) → 延长超时继续等
               否 → 位置无变化(没飞) → 判定硬件故障
          6. status=completed 或 OSD 确认到达 → 返回成功给 LLM

        面试重点:
          - 超时不是写死的固定值，是根据距离动态计算
          - 超时不等于失败——先查 OSD 实际位置再判断
          - Agent 认为失败 ≠ 物理世界停止：这是无人机 Agent 独有的难点
        """
        task_id = self._msg_id()

        # 1. 下发 MQTT 指令
        topic = f"dji/device/{self.device_id}/control/fly"
        payload = (
            f'{{"msgId":"{task_id}",'
            f'"lat":{lat:.6f},"lng":{lng:.6f},"height":{height:.1f}}}'
        )
        self._print_mqtt(topic, payload)

        # 2. 动态计算超时（基于预估距离 / 速度）
        #    生产环境: distance = haversine(current, target); eta = distance / avg_speed
        #    超时 = eta × 1.5（留 50% 余量给逆风/绕飞）
        flight_seconds = 3 + random.randint(0, 3)  # Demo 加速
        timeout = max(flight_seconds * 2, 10)       # 最少 10s

        self._mock_redis_set(task_id, _MockTaskState(
            task_id=task_id, status="started",
            progress=0, eta_seconds=flight_seconds,
        ))
        self._start_progress_simulation(task_id, flight_seconds)

        # 3. 轮询 Redis 等待到达
        for poll_count in range(1, timeout + 1):
            time.sleep(1)

            if MockExecutor.is_emergency_stopped():
                print()
                return "⛔ [EMERGENCY_STOP] 紧急停止：无人机已原地悬停，所有任务终止"

            state = self._mock_redis_get(task_id)
            if state is None:
                return f"❌ [task:{task_id}] 任务状态丢失，请重试"

            if state.status == "completed":
                # 到达
                self.current_lat = lat; self.current_lng = lng
                self.current_height = height
                self.battery -= 5 + random.randint(0, 10)
                return (f"✅ [task:{task_id}] 已到达目标点 ({lat:.6f}, {lng:.6f})，"
                        f"高度 {height:.1f}m，电量 {max(0, self.battery)}%")

            if state.status == "failed":
                return f"❌ [task:{task_id}] {state.error}"

            # 进度推前端（不推 LLM）
            if on_progress:
                on_progress(task_id, state.progress, state.status)

            eta_str = f"ETA:{state.eta_seconds}s" if state.eta_seconds else ""
            print(f"\r  ⏳ [{poll_count}s] 轮询 Redis: "
                  f"status={state.status} progress={state.progress}% {eta_str}",
                  end="", flush=True)

        # 4. 超时！不直接宣告失败——先查 OSD 实际位置兜底
        print()
        # ── 生产环境 ──
        # osd = redis.get(f"drone:osd:{device_id}:position")
        # if osd and distance_to_target(osd.lat, osd.lng, lat, lng) < 5:
        #     return "✅ 已到达目标点（OSD 位置确认，通信延迟导致状态更新滞后）"
        # if osd and position_changed(osd):
        #     continue_polling()  # 在飞但没更新状态，延长等待
        # return "❌ 飞行超时，硬件未响应。已自动悬停，请手动接管"

        # Demo 简化: 随机模拟 OSD 确认成功 / 真超时
        if random.random() < 0.6:
            self.current_lat = lat; self.current_lng = lng
            self.current_height = height
            return (f"✅ [task:{task_id}] 已到达目标点（OSD 位置确认，"
                    f"通信延迟导致 Redis 状态更新滞后）")
        return (f"❌ [task:{task_id}] 飞行超时（{timeout}s），"
                f"硬件未响应。当前位置未知，请手动检查 OSD 后重试")

    # ==================== 录像控制 ====================

    def record_for_duration(self, duration: int) -> str:
        """
        录制指定时长视频 —— 复合工具，封装 start → wait → stop。

        生产环境:
          底层硬件只有 start/stop 原子指令。
          此工具封装完整时序，LLM 只看到"录制X秒"的业务语义。

        录像不需要轮询 Redis——时长是 Agent 侧精确控制的。
        """
        task_id = self._msg_id()

        # 1. 开始录像
        topic_start = f"dji/device/{self.device_id}/camera/record/start"
        payload_start = f'{{"msgId":"{task_id}"}}'
        self._print_mqtt(topic_start, payload_start)
        self.recording = True

        # 2. 倒计时等待（Demo 加速: duration/10）
        sim_sec = max(1, min(duration // 10, 8))
        for step in range(1, sim_sec + 1):
            time.sleep(1)
            bar = "▮" * step + "▯" * (sim_sec - step)
            print(f"\r  ⏳ 录像中 [{bar}] {step}/{sim_sec}s (实际录制 {duration}s)",
                  end="", flush=True)

            # 模拟异常中断
            if random.random() < 0.03:
                print()
                self.recording = False
                return f"❌ 录像异常中断: 存储卡已满"

        print()

        # 3. 停止录像
        topic_stop = f"dji/device/{self.device_id}/camera/record/stop"
        payload_stop = f'{{"msgId":"{task_id}"}}'
        self._print_mqtt(topic_stop, payload_stop)
        self.recording = False

        return (f"✅ 录像完成，时长 {duration}s，"
                f"文件: REC_{int(time.time()) % 100000}.mp4")

    def start_recording(self) -> str:
        """开始持续录像（手动停止）。"""
        topic = f"dji/device/{self.device_id}/camera/record/start"
        payload = f'{{"msgId":"{self._msg_id()}"}}'
        self._print_mqtt(topic, payload)
        self.recording = True
        return f"✅ 开始持续录像，设备: {self.device_id}"

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
        """返航——也走 Redis 轮询模式，逻辑同 fly_to_point"""
        task_id = self._msg_id()

        topic = f"dji/device/{self.device_id}/control/return_home"
        payload = (
            f'{{"msgId":"{task_id}",'
            f'"homeLat":{self.home_lat:.6f},"homeLng":{self.home_lng:.6f}}}'
        )
        self._print_mqtt(topic, payload)

        # 模拟返航进度
        rth_seconds = 2 + random.randint(0, 2)
        self._mock_redis_set(task_id, _MockTaskState(
            task_id=task_id, status="started", progress=0, eta_seconds=rth_seconds
        ))
        self._start_progress_simulation(task_id, rth_seconds)

        for poll_count in range(1, 15):
            time.sleep(1)
            state = self._mock_redis_get(task_id)

            if state is None:
                return f"❌ 返航状态丢失，请重试"

            if state.status == "completed":
                self.current_lat = self.home_lat
                self.current_lng = self.home_lng
                self.current_height = 0.0
                self.battery -= 3 + random.randint(0, 5)
                return (f"✅ 已返航降落至起飞点 ({self.home_lat:.6f}, "
                        f"{self.home_lng:.6f})，电量 {max(0, self.battery)}%")

            if state.status == "failed":
                return f"❌ 返航异常: {state.error}"

            bar = "▮" * (poll_count % 3 + 1) + "▯" * (2 - poll_count % 3)
            print(f"\r  ⏳ 返航中 [{bar}] progress={state.progress}%",
                  end="", flush=True)

        print()
        return "❌ 返航超时，请检查无人机状态"

    # ==================== 云台控制 ====================

    def gimbal_control(self, mode: str) -> str:
        topic = f"dji/device/{self.device_id}/gimbal/control"
        angle = "0" if mode == "center" else "-90"
        desc = "回中（水平）" if mode == "center" else "垂直向下（-90°）"
        payload = f'{{"msgId":"{self._msg_id()}","pitch":{angle},"yaw":0}}'
        self._print_mqtt(topic, payload)
        self._progress("云台调整", 1)
        return f"✅ 云台已调整为 {desc}"

    # ==================== 变焦 ====================

    def set_zoom(self, factor: float) -> str:
        """
        相机变焦。factor 为变焦倍数（1x-56x）。
        对应 WVP: CameraFocalLengthSetImpl / CameraFrameZoomImpl
        """
        topic = f"dji/device/{self.device_id}/camera/zoom"
        payload = f'{{"msgId":"{self._msg_id()}","factor":{factor:.1f}}}'
        self._print_mqtt(topic, payload)
        self._progress("变焦调节", 1)
        self.zoom_level = factor
        return f"✅ 变焦已设置为 {factor:.1f}x"

    # ==================== 镜头切换 ====================

    def switch_lens(self, mode: str) -> str:
        """
        切换镜头模式。
        对应 WVP: CameraModeSwitchImpl
        """
        topic = f"dji/device/{self.device_id}/camera/lens/switch"
        payload = f'{{"msgId":"{self._msg_id()}","mode":"{mode}"}}'
        self._print_mqtt(topic, payload)
        self._progress("镜头切换", 1)
        self.lens_mode = mode

        desc = {"wide": "广角", "zoom": "变焦", "ir": "红外"}.get(mode, mode)
        return f"✅ 已切换至 {desc} 镜头"

    # ==================== 全景拍照 ====================

    def panorama_photo(self) -> str:
        """
        全景拍照。无人机保持稳定悬停约 60s，自动拍摄并合成全景图。
        对应 WVP: CameraPhotoTakeImpl (panorama mode)
        """
        topic = f"dji/device/{self.device_id}/camera/photo/panorama"
        payload = f'{{"msgId":"{self._msg_id()}","mode":"panorama"}}'
        self._print_mqtt(topic, payload)

        sim_sec = 4  # Demo 加速，实际约 60s
        for step in range(1, sim_sec + 1):
            time.sleep(1)
            progress = step * 100 // sim_sec
            bar = "▮" * step + "▯" * (sim_sec - step)
            print(f"\r  ⏳ 全景拍摄中 [{bar}] {progress}%（需保持悬停）",
                  end="", flush=True)

            if random.random() < 0.03:
                print()
                return "❌ 全景拍摄失败: 无人机晃动过大，请保持悬停后重试"

        print()
        return "✅ 全景拍照完成，照片: PANO_" + str(int(time.time()) % 100000) + ".jpg"

    # ==================== 相机状态查询 ====================

    # ==================== 曝光控制 ====================

    def set_exposure_mode(self, mode: str) -> str:
        """
        切换曝光模式。auto=自动曝光，manual=手动曝光。
        只有在 auto 模式下才能调节 ISO/快门/曝光补偿。
        对应 WVP: CameraModeSwitchImpl (exposure sub-mode)
        """
        topic = f"dji/device/{self.device_id}/camera/exposure/mode"
        payload = f'{{"msgId":"{self._msg_id()}","mode":"{mode}"}}'
        self._print_mqtt(topic, payload)
        self._progress("曝光模式切换", 1)
        self.exposure_mode = mode
        desc = "自动曝光" if mode == "auto" else "手动曝光"
        return f"✅ 已切换为 {desc}"

    def set_iso(self, value: int) -> str:
        """设置 ISO（仅在自动曝光模式下可用）"""
        topic = f"dji/device/{self.device_id}/camera/iso"
        payload = f'{{"msgId":"{self._msg_id()}","iso":{value}}}'
        self._print_mqtt(topic, payload)
        self._progress("ISO 调节", 1)
        self.iso = value
        return f"✅ ISO 已设置为 {value}"

    def set_shutter_speed(self, speed: str) -> str:
        """设置快门速度（仅在自动曝光模式下可用），如 '1/100' '1/500'"""
        topic = f"dji/device/{self.device_id}/camera/shutter"
        payload = f'{{"msgId":"{self._msg_id()}","shutter":"{speed}"}}'
        self._print_mqtt(topic, payload)
        self._progress("快门调节", 1)
        self.shutter_speed = speed
        return f"✅ 快门已设置为 {speed}s"

    def set_ev_compensation(self, ev: float) -> str:
        """设置曝光补偿（仅在自动曝光模式下可用），范围 -3.0 ~ +3.0"""
        topic = f"dji/device/{self.device_id}/camera/ev"
        payload = f'{{"msgId":"{self._msg_id()}","ev":{ev:.1f}}}'
        self._print_mqtt(topic, payload)
        self._progress("曝光补偿调节", 1)
        self.ev_compensation = ev
        return f"✅ 曝光补偿已设置为 {ev:+.1f}EV"

    # ==================== 相机状态查询 ====================

    def get_camera_status(self) -> str:
        """查询相机当前参数"""
        lens = {"wide": "广角", "zoom": "变焦", "ir": "红外"}.get(
            self.lens_mode, self.lens_mode
        )
        exp = "自动曝光" if self.exposure_mode == "auto" else "手动曝光"
        return (
            f"📷 相机状态 [{self.device_id}]:\n"
            f"  镜头: {lens}\n"
            f"  变焦: {self.zoom_level:.1f}x\n"
            f"  曝光模式: {exp}\n"
            f"  ISO: {self.iso}\n"
            f"  快门: {self.shutter_speed}s\n"
            f"  曝光补偿: {self.ev_compensation:+.1f}EV\n"
            f"  录像: {'进行中' if self.recording else '待机'}\n"
            f"  存储: 可用 128GB / 256GB"
        )

    # ==================== 状态查询 ====================

    def get_status(self) -> str:
        mode = "空中悬停" if self.current_height > 0 else "地面待命"
        rec = "录像中" if self.recording else "未录像"
        # 生产环境: 从 Redis 读取实时 OSD 数据
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
                print(f"\r  ⏳ {action} [{bar}] {i}/{seconds}s",
                      end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\n  ⚠️ 操作被中断")
            raise
