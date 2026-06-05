"""
安全校验层 —— 100% 硬编码规则

设计原则（面试重点）：
- LLM 负责意图理解 —— 把"飞到5号楼"变成 GPS 坐标
- SafetyGate 负责安全决策 —— 这个坐标能不能飞、高度合不合法
- 两个角色绝对不能混淆。安全规则必须是确定性的、可审计的、秒级响应的。
- LLM 的输出本质上是概率性的，安全层的决策永远不能交给统计模型。

生产化路径：
- 规则可从数据库/配置文件加载，甚至引入规则引擎
- 但核心原则不变 —— 安全层永远是硬编码逻辑，不经过 LLM 判断
"""
import math
from dataclasses import dataclass


@dataclass
class SafetyResult:
    passed: bool
    reason: str
    warning: bool = False


class SafetyGate:
    # 中国境内大致经纬度范围
    MIN_LAT, MAX_LAT = 18.0, 54.0
    MIN_LNG, MAX_LNG = 73.0, 135.0

    # 飞行高度限制（米）
    MAX_HEIGHT = 120.0
    MIN_HEIGHT = 10.0

    def __init__(self, home_lat: float, home_lng: float):
        self.home_lat = home_lat
        self.home_lng = home_lng

    def validate_fly(self, lat: float, lng: float, height: float) -> SafetyResult:
        """
        校验飞行指令。
        规则 1: 坐标必须在合理范围（防止 LLM 幻觉编造境外坐标）
        规则 2: 高度必须在安全范围（法规硬约束，LLM 可能产生幻觉）
        规则 3: 电量预估（需要精确计算，LLM 不擅长数学）
        """
        if not (self.MIN_LAT <= lat <= self.MAX_LAT):
            return SafetyResult(False, f"纬度 {lat} 不在境内范围 [{self.MIN_LAT}, {self.MAX_LAT}]")
        if not (self.MIN_LNG <= lng <= self.MAX_LNG):
            return SafetyResult(False, f"经度 {lng} 不在境内范围 [{self.MIN_LNG}, {self.MAX_LNG}]")

        if height > self.MAX_HEIGHT:
            return SafetyResult(False, f"高度 {height}m 超过限制 {self.MAX_HEIGHT}m")
        if height < self.MIN_HEIGHT:
            return SafetyResult(False, f"高度 {height}m 低于最低安全高度 {self.MIN_HEIGHT}m")

        # 电量预估（简化：每公里耗电 3%，+10% 余量用于返航悬停）
        distance_km = self._estimate_distance(lat, lng)
        consumption = distance_km * 3.0 + 10
        if consumption > 70:
            return SafetyResult(
                True,
                f"预估耗电 {consumption:.0f}%，电量可能不足，建议确认后执行",
                warning=True
            )

        print(f"  ✅ 安全校验通过 | 距离: {distance_km:.1f}km | 预估耗电: {consumption:.0f}%")
        return SafetyResult(True, "ok")

    def validate_return_home(self) -> SafetyResult:
        """校验返航指令。生产环境会从 Redis/OSD 读取实时数据做更精确判断。"""
        return SafetyResult(True, "ok")

    def _estimate_distance(self, target_lat: float, target_lng: float) -> float:
        """球面余弦公式计算直线距离（km）"""
        dlat = math.radians(target_lat - self.home_lat)
        dlng = math.radians(target_lng - self.home_lng)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(self.home_lat))
             * math.cos(math.radians(target_lat))
             * math.sin(dlng / 2) ** 2)
        return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
