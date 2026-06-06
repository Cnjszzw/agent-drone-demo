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

    # ==================== Redis 模拟接口 ====================

    def _mock_redis_get(self, task_id: str) -> _MockTaskState | None:
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

                # 模拟偶发故障（3% 概率，只在中途触发）
                if elapsed > total_seconds * 0.3 and random.random() < 0.03:
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
        指点飞行 —— 内部封装 Redis 轮询等待。

        生产环境实际流程:
          1. POST /api/drone/control/fly → Java 下发 MQTT → 返回 task_id
          2. 嵌入式 → MQTT 进度消息 → Java 消费 → 写入 Redis
          3. Python redis-py: GET drone:task:{task_id}:status（1s 间隔）
          4. 进度仅推前端展示 → LLM 不感知中间状态
          5. status=completed/failed 时返回结果给 LLM

        轮询参数:
          间隔: 1s（对 2-3 分钟飞行，60-180 次 GET，localhost Redis 零负载）
          超时: 180s（3 分钟，超过则判定异常）

        LLM 视角: fly_to_point() = 一次函数调用。返回 = 到达/失败/超时。
        """
        task_id = self._msg_id()

        # 1. 下发 MQTT 指令
        topic = f"dji/device/{self.device_id}/control/fly"
        payload = (
            f'{{"msgId":"{task_id}",'
            f'"lat":{lat:.6f},"lng":{lng:.6f},"height":{height:.1f}}}'
        )
        self._print_mqtt(topic, payload)

        # 2. 初始化模拟 Redis 状态
        flight_seconds = 3 + random.randint(0, 3)  # Demo 加速: 3-6s
        # 生产环境: 真实飞行时间 60-180s，由硬件决定
        self._mock_redis_set(task_id, _MockTaskState(
            task_id=task_id,
            status="started",
            progress=0,
            eta_seconds=flight_seconds,
        ))

        # 3. 启动后台模拟进度更新
        #    生产环境: 嵌入式硬件 → MQTT → Java → Redis（无需此行代码）
        self._start_progress_simulation(task_id, flight_seconds)

        # 4. 轮询 Redis 等待到达
        #    生产环境: while True: state = redis.get(...)
        max_wait = 30  # Demo 最多等 30s（生产环境改为 180）
        for poll_count in range(1, max_wait + 1):
            time.sleep(1)

            # ── 生产环境替换为 ──
            # raw = redis.get(f"drone:task:{task_id}:status")
            # state = json.loads(raw) if raw else None
            state = self._mock_redis_get(task_id)

            if state is None:
                return f"❌ [task:{task_id}] 任务状态丢失，请重试"

            if state.status == "completed":
                # 更新本地模拟状态
                self.current_lat = lat
                self.current_lng = lng
                self.current_height = height
                self.battery -= 5 + random.randint(0, 10)

                return (f"✅ [task:{task_id}] 已到达目标点 ({lat:.6f}, {lng:.6f})，"
                        f"高度 {height:.1f}m，电量 {max(0, self.battery)}%")

            if state.status == "failed":
                return f"❌ [task:{task_id}] {state.error}"

            # 进度推前端展示（不推 LLM）
            if on_progress:
                on_progress(task_id, state.progress, state.status)

            # 终端日志（生产环境去掉，进度走前端）
            eta_str = f"ETA:{state.eta_seconds}s" if state.eta_seconds else ""
            print(f"\r  ⏳ [{poll_count}s] 轮询 Redis: "
                  f"status={state.status} progress={state.progress}% {eta_str}",
                  end="", flush=True)

        # 超时
        print()
        return f"❌ [task:{task_id}] 飞行超时（{max_wait}s），请检查无人机状态"

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
