"""
Smart Co-Pilot — Hybrid Decision Engine
========================================
Mimari: Rule-Based (Phase 1) → ML Clustering (Phase 2) → Optimization (Phase 3)
Çıktı: FastAPI-ready JSON — Dashboard ve gerçek donanımla plug-and-play uyumlu.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AlertLevel(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class ScenarioType(str, Enum):
    NORMAL = "NORMAL"
    HEATWAVE = "HEATWAVE"
    DROUGHT = "DROUGHT"
    COLD_STRESS = "COLD_STRESS"
    HARDWARE_FAULT = "HARDWARE_FAULT"


ALERT_SEVERITY = {
    AlertLevel.OK: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.CRITICAL: 2,
    AlertLevel.EMERGENCY: 3,
}


@dataclass
class SensorReading:
    timestamp: str
    temperature: float
    humidity: float
    water_level: float
    nitrogen: int
    phosphorus: int
    potassium: int
    plant_type: str = "lettuce"
    fan_on: int = 0
    watering_pump_on: int = 0
    water_pump_on: int = 0


@dataclass
class ActuatorCommand:
    fan: bool = False
    watering_pump: bool = False
    water_pump: bool = False

    def to_dict(self):
        return {
            "fan": self.fan,
            "watering_pump": self.watering_pump,
            "water_pump": self.water_pump,
        }


@dataclass
class DecisionResult:
    timestamp: str
    alert_level: AlertLevel
    scenario: ScenarioType
    sensor_data: dict
    actuator_command: dict
    rule_based_flags: list[str]
    ml_cluster: Optional[int]
    ml_trend: Optional[dict]
    optimization: Optional[dict]
    decision_comment: str
    user_view: dict
    literature_refs: list[str]
    user_action_required: bool
    confidence_score: float

    def to_json(self):
        d = asdict(self)
        d["alert_level"] = self.alert_level.value
        d["scenario"] = self.scenario.value
        return json.dumps(d, ensure_ascii=False, indent=2)


THRESHOLDS = {
    "temp_ok": (15.0, 30.0),
    "temp_warn_high": 30.0,
    "temp_critical": 38.0,
    "temp_emergency": 42.0,
    "temp_warn_low": 10.0,
    "temp_critical_low": 5.0,
    "humidity_ok": (50.0, 75.0),
    "humidity_warn_low": 40.0,
    "humidity_critical_low": 20.0,
    "humidity_warn_high": 80.0,
    "water_level_ok": 50.0,
    "water_level_warn": 30.0,
    "water_level_critical": 10.0,
    "water_level_empty": 2.0,
}

PLANT_PROFILES = {
    "lettuce": {
        "label": "Marul",
        "temp_warn_high": 30.0,
        "humidity_warn_low": 40.0,
        "water_level_warn": 30.0,
    },
    "basil": {
        "label": "Fesleğen",
        "temp_warn_high": 32.0,
        "humidity_warn_low": 42.0,
        "water_level_warn": 32.0,
    },
    "spinach": {
        "label": "Ispanak",
        "temp_warn_high": 28.0,
        "humidity_warn_low": 45.0,
        "water_level_warn": 34.0,
    },
}

LITERATURE = {
    "heatwave": [
        "Kozai et al. (2022) — Plant Factory: An Indoor Vertical Farming System, s.142",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
        "ASHRAE (2021) — Thermal Guidelines for Controlled Environment Agriculture",
    ],
    "drought": [
        "Jones (2014) — Plants and Microclimate, Cambridge UP, s.88",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
        "Savvas & Gruda (2018) — Hydroponic and Aeroponic Systems, Scientia Horticulturae",
    ],
    "cold_stress": [
        "Kozai et al. (2022) — Plant Factory: An Indoor Vertical Farming System",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
    ],
    "hardware_fault": [
        "Gubbi et al. (2013) — IoT for Smart Cities: Vision & Challenges, Future Generation CS",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
    ],
    "normal": [
        "Kozai et al. (2022) — Plant Factory: An Indoor Vertical Farming System",
    ],
}


def _raise_alert(current: AlertLevel, target: AlertLevel) -> AlertLevel:
    if ALERT_SEVERITY[target] > ALERT_SEVERITY[current]:
        return target
    return current


def _thresholds_for_plant(plant_type: str) -> dict:
    profile = PLANT_PROFILES.get(plant_type, PLANT_PROFILES["lettuce"])
    th = dict(THRESHOLDS)
    th["temp_warn_high"] = profile["temp_warn_high"]
    th["humidity_warn_low"] = profile["humidity_warn_low"]
    th["water_level_warn"] = profile["water_level_warn"]
    return th


def run_rule_based(reading: SensorReading) -> tuple[AlertLevel, ScenarioType, ActuatorCommand, list[str], float]:
    flags = []
    alert = AlertLevel.OK
    scenario = ScenarioType.NORMAL
    cmd = ActuatorCommand()
    confidence = 1.0

    th = _thresholds_for_plant(reading.plant_type)
    t = reading.temperature
    h = reading.humidity
    wl = reading.water_level

    if t >= th["temp_emergency"]:
        flags.append(f"EMERGENCY: Sıcaklık {t}°C — Bitki ölüm riski!")
        alert = AlertLevel.EMERGENCY
        scenario = ScenarioType.HEATWAVE
        cmd.fan = True
    elif t >= th["temp_critical"]:
        flags.append(f"CRITICAL: Sıcaklık {t}°C — Kritik ısı stresi")
        alert = AlertLevel.CRITICAL
        scenario = ScenarioType.HEATWAVE
        cmd.fan = True
    elif t >= th["temp_warn_high"]:
        flags.append(f"WARNING: Sıcaklık {t}°C — Optimal aralık aşıldı")
        alert = AlertLevel.WARNING
        cmd.fan = True
    elif t <= th["temp_critical_low"]:
        flags.append(f"CRITICAL: Sıcaklık {t}°C — Dondurma riski!")
        alert = AlertLevel.CRITICAL
        scenario = ScenarioType.COLD_STRESS
    elif t <= th["temp_warn_low"]:
        flags.append(f"WARNING: Sıcaklık {t}°C — Soğuk stresi başlıyor")
        alert = _raise_alert(alert, AlertLevel.WARNING)
        scenario = ScenarioType.COLD_STRESS

    if h <= th["humidity_critical_low"]:
        flags.append(f"CRITICAL: Nem %{h} — Ciddi kuraklık stresi (VPD çok yüksek)")
        alert = _raise_alert(alert, AlertLevel.CRITICAL)
        scenario = ScenarioType.DROUGHT
        cmd.watering_pump = True
    elif h <= th["humidity_warn_low"]:
        flags.append(f"WARNING: Nem %{h} — Düşük nem, VPD optimal değil")
        alert = _raise_alert(alert, AlertLevel.WARNING)
        scenario = ScenarioType.DROUGHT if scenario == ScenarioType.NORMAL else scenario
        cmd.watering_pump = True
    elif h >= th["humidity_warn_high"]:
        flags.append(f"WARNING: Nem %{h} — Aşırı nem, hastalık riski")
        alert = _raise_alert(alert, AlertLevel.WARNING)
        cmd.fan = True

    if wl <= th["water_level_empty"]:
        flags.append(f"EMERGENCY: Su seviyesi %{wl} — Pompa koruması devrede, sulama DURDU")
        alert = AlertLevel.EMERGENCY
        scenario = ScenarioType.DROUGHT
        cmd.watering_pump = False
        cmd.water_pump = False
    elif wl <= th["water_level_critical"]:
        flags.append(f"CRITICAL: Su seviyesi %{wl} — Tank neredeyse boş")
        alert = _raise_alert(alert, AlertLevel.CRITICAL)
        scenario = ScenarioType.DROUGHT
        cmd.water_pump = True
    elif wl <= th["water_level_warn"]:
        flags.append(f"WARNING: Su seviyesi %{wl} — Yenileme önerisi")
        alert = _raise_alert(alert, AlertLevel.WARNING)

    if reading.watering_pump_on == 1 and h <= th["humidity_warn_low"]:
        flags.append("FAULT DETECTED: Sulama pompası açık ama nem artmıyor — Olası boru/pompa arızası!")
        scenario = ScenarioType.HARDWARE_FAULT
        alert = _raise_alert(alert, AlertLevel.CRITICAL)
        confidence = 0.75

    if not flags:
        flags.append("OK: Tüm parametreler normal aralıkta")

    return alert, scenario, cmd, flags, confidence


class MLEngine:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.fitted = False
        self.cluster_labels = {
            0: "Optimal Koşullar",
            1: "Isı Stresi",
            2: "Kuraklık Stresi",
            3: "Kritik Durum",
        }
        self._window: list[dict] = []
        self.window_size = 10

    def fit(self, df: pd.DataFrame):
        features = df[["temperature", "humidity", "water_level"]].values
        scaled = self.scaler.fit_transform(features)
        self.kmeans.fit(scaled)
        self.fitted = True
        self._assign_cluster_semantics()

    def _assign_cluster_semantics(self):
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        for i, center in enumerate(centers):
            temp, hum, wl = center[0], center[1], center[2]
            if temp > 38:
                self.cluster_labels[i] = "Isı Stresi"
            elif hum < 30 or wl < 20:
                self.cluster_labels[i] = "Kuraklık Stresi"
            elif temp < 10:
                self.cluster_labels[i] = "Soğuk Stresi"
            elif 15 <= temp <= 30 and 50 <= hum <= 75 and wl >= 50:
                self.cluster_labels[i] = "Optimal Koşullar"
            else:
                self.cluster_labels[i] = "Stresli / Geçiş"

    def _calculate_trend(self, values: list[float], label: str) -> dict:
        if len(values) < 3:
            return {
                "direction": "belirsiz",
                "rate_per_reading": 0.0,
                "current": round(values[-1], 1) if values else 0,
                "eta_to_critical": "Yeterli veri yok",
            }

        mid = len(values) // 2
        recent_avg = sum(values[mid:]) / len(values[mid:])
        older_avg = sum(values[:mid]) / len(values[:mid])
        rate = recent_avg - older_avg

        thresholds = {
            "temperature": {"critical": 38.0, "direction": "up"},
            "humidity": {"critical": 20.0, "direction": "down"},
            "water_level": {"critical": 10.0, "direction": "down"},
        }

        current = values[-1]
        t = thresholds.get(label, {})

        if abs(rate) < 0.3:
            direction = "stabil"
        elif rate > 0:
            direction = "artıyor"
        else:
            direction = "azalıyor"

        eta = "Kritik eşik riski yok"
        if t and abs(rate) > 0.1:
            critical = t["critical"]
            if t["direction"] == "up" and direction == "artıyor" and current < critical:
                steps = abs((critical - current) / rate)
                eta = f"~{int(steps * 5)} dakikada kritik eşiğe ulaşabilir"
            elif t["direction"] == "down" and direction == "azalıyor" and current > critical:
                steps = abs((current - critical) / abs(rate))
                eta = f"~{int(steps * 5)} dakikada kritik eşiğe ulaşabilir"

        return {
            "direction": direction,
            "rate_per_reading": round(rate, 2),
            "current": round(current, 1),
            "eta_to_critical": eta,
        }

    def predict(self, reading: SensorReading) -> dict:
        if not self.fitted:
            return {"cluster": None, "cluster_label": "Model eğitilmedi", "trend": None}

        self._window.append(
            {
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "water_level": reading.water_level,
            }
        )
        if len(self._window) > self.window_size:
            self._window.pop(0)

        features = np.array([[reading.temperature, reading.humidity, reading.water_level]])
        scaled = self.scaler.transform(features)
        cluster_id = int(self.kmeans.predict(scaled)[0])

        trend = {}
        for col in ["temperature", "humidity", "water_level"]:
            values = [r[col] for r in self._window]
            trend[col] = self._calculate_trend(values, col)

        return {
            "cluster": cluster_id,
            "cluster_label": self.cluster_labels.get(cluster_id, "Bilinmiyor"),
            "trend": trend,
        }


def _estimate_next_state(reading: SensorReading, cmd: ActuatorCommand) -> dict:
    temp = reading.temperature - (1.2 if cmd.fan else 0.0) + (0.2 if not cmd.fan and reading.temperature > 32 else 0.0)
    # What-if model calibration:
    # Sulama kısa vadede nemi anlamlı toparlar; fan etkisi nemi azaltır ama baskın olmamalı.
    humidity = reading.humidity + (3.5 if cmd.watering_pump else 0.0) - (0.8 if cmd.fan else 0.0)
    # Sulama tank seviyesini düşürür ancak her adımda aşırı tüketim varsayımı yapılmaz.
    water_level = reading.water_level - (0.8 if cmd.watering_pump else 0.0) + (3.0 if cmd.water_pump else 0.0)
    return {
        "temperature": max(-10.0, min(60.0, temp)),
        "humidity": max(0.0, min(100.0, humidity)),
        "water_level": max(0.0, min(100.0, water_level)),
    }


def _safety_penalty(next_state: dict, th: dict) -> float:
    penalty = 0.0
    penalty += max(0.0, next_state["temperature"] - th["temp_warn_high"]) * 1.4
    penalty += max(0.0, th["temp_warn_low"] - next_state["temperature"]) * 1.4
    penalty += max(0.0, th["humidity_warn_low"] - next_state["humidity"]) * 1.1
    penalty += max(0.0, next_state["humidity"] - th["humidity_warn_high"]) * 0.7
    penalty += max(0.0, th["water_level_warn"] - next_state["water_level"]) * 1.2
    return penalty


def optimize_actuator_plan(
    reading: SensorReading,
    base_command: ActuatorCommand,
    alert: AlertLevel,
    scenario: ScenarioType,
) -> dict:
    """
    Phase 4 optimization layer.

    Objective function (minimize):
      J = w_safety*Risk + w_energy*Energy + w_water*WaterUse + w_switch*SwitchCost + w_user*ManualLoad

    Brute-force over 2^3 actuator combinations with hard safety constraints.
    """
    weights = {
        "safety": 5.0,
        "energy": 1.1,
        "water": 0.7,
        "switch": 0.6,
        "user": 0.8,
    }
    th = _thresholds_for_plant(reading.plant_type)

    base = base_command.to_dict()
    best = None

    for fan, watering_pump, water_pump in product([False, True], repeat=3):
        cmd = ActuatorCommand(fan=fan, watering_pump=watering_pump, water_pump=water_pump)

        if reading.water_level <= th["water_level_empty"] and (watering_pump or water_pump):
            continue
        if reading.temperature >= th["temp_critical"] and not fan:
            continue
        if reading.humidity <= th["humidity_critical_low"] and not watering_pump:
            continue

        next_state = _estimate_next_state(reading, cmd)
        risk = _safety_penalty(next_state, th)
        energy = 1.0 * fan + 1.2 * watering_pump + 1.4 * water_pump
        water_use = 1.5 * watering_pump + 0.8 * water_pump
        switch_cost = float(base["fan"] != fan) + float(base["watering_pump"] != watering_pump) + float(base["water_pump"] != water_pump)
        manual_load = 1.0 if (ALERT_SEVERITY[alert] >= 2 and not fan and reading.temperature > th["temp_warn_high"]) else 0.0

        objective = (
            weights["safety"] * risk
            + weights["energy"] * energy
            + weights["water"] * water_use
            + weights["switch"] * switch_cost
            + weights["user"] * manual_load
        )

        row = {
            "command": cmd.to_dict(),
            "objective": round(objective, 3),
            "components": {
                "risk": round(risk, 3),
                "energy": round(energy, 3),
                "water_use": round(water_use, 3),
                "switch_cost": round(switch_cost, 3),
                "manual_load": round(manual_load, 3),
            },
            "predicted_next_state": {k: round(v, 2) for k, v in next_state.items()},
        }

        if best is None or row["objective"] < best["objective"]:
            best = row

    if best is None:
        best = {
            "command": base,
            "objective": 999.0,
            "components": {
                "risk": 999.0,
                "energy": 0.0,
                "water_use": 0.0,
                "switch_cost": 0.0,
                "manual_load": 0.0,
            },
            "predicted_next_state": {
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "water_level": reading.water_level,
            },
        }

    return {
        "objective_function": "Güvenlik riski, enerji, su kullanımı ve anahtarlama maliyeti dengelenerek en uygun komut seçilir.",
        "scenario": scenario.value,
        "plant_type": reading.plant_type,
        "plant_label": PLANT_PROFILES.get(reading.plant_type, PLANT_PROFILES["lettuce"])["label"],
        "recommended_command": best["command"],
        "objective_score": best["objective"],
        "cost_breakdown": best["components"],
        "predicted_next_state": best["predicted_next_state"],
    }


def _friendly_actuator_name(key: str) -> str:
    mapping = {
        "fan": "fan",
        "watering_pump": "sulama pompası",
        "water_pump": "su doldurma pompası",
    }
    return mapping.get(key, key)


def _delta_to_phrase(delta: float, metric: str) -> str:
    if abs(delta) < 0.1:
        return f"{metric} neredeyse sabit kalır"
    if delta > 0:
        return f"{metric} biraz artar"
    return f"{metric} biraz azalır"


def _humanize_flag_text(flag: str) -> str:
    cleaned = flag
    for prefix in ("EMERGENCY:", "CRITICAL:", "WARNING:", "FAULT DETECTED:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned


def _what_if_warnings(reading: SensorReading, cmd: ActuatorCommand) -> list[str]:
    current_alert, current_scenario, _, _, _ = run_rule_based(reading)
    warnings = []

    if current_scenario == ScenarioType.COLD_STRESS and cmd.fan:
        warnings.append("Soğuk stresinde fan açmak ortamı daha da soğutabilir ve toparlanmayı geciktirebilir.")
    if current_scenario == ScenarioType.DROUGHT and cmd.fan and not cmd.watering_pump:
        warnings.append("Kuraklık riskinde fan tek başına nemi daha da düşürebilir.")
    if current_scenario == ScenarioType.DROUGHT and cmd.watering_pump and not cmd.water_pump and reading.water_level <= _thresholds_for_plant(reading.plant_type)["water_level_warn"]:
        warnings.append("Sulama pompası açıkken su takviyesi kapalıysa tank seviyesi daha hızlı düşebilir.")
    if current_scenario == ScenarioType.HEATWAVE and not cmd.fan and ALERT_SEVERITY[current_alert] >= 2:
        warnings.append("Isı stresinde fan kapalı kalırsa sıcaklık güvenli aralığa daha yavaş döner.")

    return warnings


def build_decision_comment(
    reading: SensorReading,
    alert: AlertLevel,
    scenario: ScenarioType,
    flags: list[str],
    actuator: ActuatorCommand,
    trend: Optional[dict],
    optimization: Optional[dict],
) -> str:
    """Yalnızca rule-based + decision engine çıktısıyla sade yorum üretir."""
    severity_text = {
        AlertLevel.OK: "Bitkilerin durumu dengeli görünüyor.",
        AlertLevel.WARNING: "Bitkilerde küçük dengesizlikler var, erken müdahale iyi olur.",
        AlertLevel.CRITICAL: "Bitkiler stres altında, hızlı müdahale gerekiyor.",
        AlertLevel.EMERGENCY: "Acil risk var, sistemi hemen kontrol etmelisin.",
    }
    scenario_text = {
        ScenarioType.NORMAL: "Normal çalışma",
        ScenarioType.HEATWAVE: "Isı stresi",
        ScenarioType.DROUGHT: "Kuraklık riski",
        ScenarioType.COLD_STRESS: "Soğuk stresi",
        ScenarioType.HARDWARE_FAULT: "Donanım arızası şüphesi",
    }
    active_cmd = [_friendly_actuator_name(k) for k, v in actuator.to_dict().items() if v]
    command_text = ", ".join(active_cmd) if active_cmd else "şu an ek bir cihaz açmaya gerek yok"

    trend_alerts = []
    if trend:
        for label, data in trend.items():
            eta = data.get("eta_to_critical", "")
            if "dakikada kritik eşiğe ulaşabilir" in eta:
                metric_name = {
                    "temperature": "Sıcaklık",
                    "humidity": "Nem",
                    "water_level": "Su seviyesi",
                }.get(label, label)
                trend_alerts.append(f"{metric_name} için {eta}")
    trend_text = "; ".join(trend_alerts) if trend_alerts else "Kısa vadede kritik bir kötüleşme sinyali yok."

    opt_text = "Tahmini etki bilgisi henüz oluşmadı."
    if optimization:
        ns = optimization.get("predicted_next_state", {})
        t_delta = float(ns.get("temperature", reading.temperature)) - reading.temperature
        h_delta = float(ns.get("humidity", reading.humidity)) - reading.humidity
        wl_delta = float(ns.get("water_level", reading.water_level)) - reading.water_level
        opt_text = (
            f"Bu ayarla {_delta_to_phrase(t_delta, 'sıcaklık')}, "
            f"{_delta_to_phrase(h_delta, 'nem')} ve {_delta_to_phrase(wl_delta, 'su seviyesi')}."
        )

    approval = "Bu adım için kullanıcı onayı gerekli." if alert in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY) else "İstersen bu ayarı uygulayabilirsin."
    scenario_label = scenario_text.get(scenario, scenario.value)
    issue_text = " • ".join(_humanize_flag_text(f) for f in flags[:2]) if flags else "Özel bir alarm yok."

    return (
        f"{severity_text[alert]} Şu anki durum: {scenario_label}. "
        f"Önerilen müdahale: {command_text}. "
        f"Gözlenen sorunlar: {issue_text}. Trend: {trend_text}. "
        f"{opt_text} {approval}"
    )


def build_user_view(
    reading: SensorReading,
    alert: AlertLevel,
    scenario: ScenarioType,
    actuator: ActuatorCommand,
    optimization: Optional[dict],
    comment: str,
) -> dict:
    status_map = {
        AlertLevel.OK: "İyi",
        AlertLevel.WARNING: "Dikkat",
        AlertLevel.CRITICAL: "Kritik",
        AlertLevel.EMERGENCY: "Acil",
    }
    scenario_map = {
        ScenarioType.NORMAL: "Normal denge",
        ScenarioType.HEATWAVE: "Isı stresi",
        ScenarioType.DROUGHT: "Kuraklık riski",
        ScenarioType.COLD_STRESS: "Soğuk stresi",
        ScenarioType.HARDWARE_FAULT: "Donanım arızası şüphesi",
    }
    action_recommendations = []
    if actuator.watering_pump:
        action_recommendations.append(
            {
                "priority": 1,
                "type": "watering",
                "title": "Sulamayı başlat",
                "detail": "Kök bölgesindeki kuruma riskini azaltmak için sulama aktif olmalı.",
                "icon": "watering-plant",
            }
        )
    if actuator.fan:
        action_recommendations.append(
            {
                "priority": 2,
                "type": "cooling",
                "title": "Hava akışını artır",
                "detail": "Sıcaklık ve nem dengesini toparlamak için fan desteği öneriliyor.",
                "icon": "fan-breeze",
            }
        )
    if actuator.water_pump:
        action_recommendations.append(
            {
                "priority": 3,
                "type": "refill",
                "title": "Su tankını destekle",
                "detail": "Su seviyesi düşüşünü durdurmak için ana depoya su takviyesi yap.",
                "icon": "water-refill",
            }
        )
    if not action_recommendations:
        if scenario == ScenarioType.DROUGHT:
            action_recommendations.append(
                {
                    "priority": 1,
                    "type": "watering",
                    "title": "Sulamayı başlat",
                    "detail": "Kuraklık riskinde nemi toparlamak için sulama öncelikli olmalı.",
                    "icon": "watering-plant",
                }
            )
        elif scenario == ScenarioType.COLD_STRESS:
            action_recommendations.append(
                {
                    "priority": 1,
                    "type": "warming",
                    "title": "Ortamı ılımlaştır",
                    "detail": "Soğuk stresini azaltmak için ısı kaybını düşür ve ortamı stabilize et.",
                    "icon": "cold-plant",
                }
            )
        elif scenario == ScenarioType.HARDWARE_FAULT:
            action_recommendations.append(
                {
                    "priority": 1,
                    "type": "maintenance",
                    "title": "Pompa hattını kontrol et",
                    "detail": "Sensör ve pompa davranışı tutarsız; fiziksel hat/pompa arızası olabilir.",
                    "icon": "tool-check",
                }
            )
        else:
            action_recommendations.append(
                {
                    "priority": 1,
                    "type": "monitor",
                    "title": "Mevcut düzeni koru",
                    "detail": "Şu an kritik müdahale gerekmiyor, rutin izleme yeterli.",
                    "icon": "healthy-leaf",
                }
            )

    action_recommendations.sort(key=lambda x: x["priority"])
    actions = [_friendly_actuator_name(k) for k, v in actuator.to_dict().items() if v]
    if not actions:
        actions = ["Şimdilik cihaz değişimi gerekmiyor"]

    predicted = optimization.get("predicted_next_state", {}) if optimization else {}
    predicted_state = {
        "temperature": round(float(predicted.get("temperature", reading.temperature)), 1),
        "humidity": round(float(predicted.get("humidity", reading.humidity)), 1),
        "water_level": round(float(predicted.get("water_level", reading.water_level)), 1),
    }

    return {
        "status_label": status_map.get(alert, alert.value),
        "scenario_label": scenario_map.get(scenario, scenario.value),
        "actions": actions,
        "action_recommendations": action_recommendations,
        "highlight_action": action_recommendations[0],
        "plant_label": PLANT_PROFILES.get(reading.plant_type, PLANT_PROFILES["lettuce"])["label"],
        "plant_type": reading.plant_type,
        "predicted_state": predicted_state,
        "summary": comment,
    }


def evaluate_what_if(
    reading: SensorReading,
    fan: bool,
    watering_pump: bool,
    water_pump: bool,
) -> dict:
    cmd = ActuatorCommand(fan=fan, watering_pump=watering_pump, water_pump=water_pump)
    predicted_state = _estimate_next_state(reading, cmd)

    # Tek adım yerine kısa ufuklu projeksiyon: kullanıcı etkisini birkaç döngü sonrası görsün.
    horizon_steps = 8
    projected = {
        "temperature": reading.temperature,
        "humidity": reading.humidity,
        "water_level": reading.water_level,
    }
    for _ in range(horizon_steps):
        projected = _estimate_next_state(
            SensorReading(
                timestamp=reading.timestamp,
                temperature=projected["temperature"],
                humidity=projected["humidity"],
                water_level=projected["water_level"],
                nitrogen=reading.nitrogen,
                phosphorus=reading.phosphorus,
                potassium=reading.potassium,
                plant_type=reading.plant_type,
                fan_on=0,
                watering_pump_on=0,
                water_pump_on=0,
            ),
            cmd,
        )

    future_reading = SensorReading(
        timestamp=reading.timestamp,
        temperature=projected["temperature"],
        humidity=projected["humidity"],
        water_level=projected["water_level"],
        nitrogen=reading.nitrogen,
        phosphorus=reading.phosphorus,
        potassium=reading.potassium,
        plant_type=reading.plant_type,
        # What-if'te arıza dedeksiyonu değil, çevresel etkiler simüle edilir.
        fan_on=0,
        watering_pump_on=0,
        water_pump_on=0,
    )
    future_alert, future_scenario, _, _, _ = run_rule_based(future_reading)
    warnings = _what_if_warnings(reading, cmd)

    return {
        "selected_command": cmd.to_dict(),
        "predicted_state": {
            "temperature": round(predicted_state["temperature"], 1),
            "humidity": round(predicted_state["humidity"], 1),
            "water_level": round(predicted_state["water_level"], 1),
        },
        "projected_state": {
            "temperature": round(projected["temperature"], 1),
            "humidity": round(projected["humidity"], 1),
            "water_level": round(projected["water_level"], 1),
        },
        "projection_steps": horizon_steps,
        "expected_alert_level": future_alert.value,
        "expected_scenario": future_scenario.value,
        "warnings": warnings,
        "summary": build_decision_comment(
            reading=reading,
            alert=future_alert,
            scenario=future_scenario,
            flags=[],
            actuator=cmd,
            trend=None,
            optimization={
                "predicted_next_state": predicted_state,
            },
        ),
    }


class HybridDecisionEngine:
    def __init__(self):
        self.ml_engine = MLEngine(n_clusters=4)
        self._history: list[SensorReading] = []

    def fit(self, df: pd.DataFrame):
        col_map = {"tempreature": "temperature"}
        df = df.rename(columns=col_map)

        required = ["temperature", "humidity", "water_level"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Eksik kolonlar: {missing}")

        self.ml_engine.fit(df.dropna(subset=required))
        print(f"[ML] Model {len(df)} satır veriyle eğitildi. Kümeler: {self.ml_engine.cluster_labels}")

    def decide(self, reading: SensorReading) -> DecisionResult:
        self._history.append(reading)

        alert, scenario, actuator, flags, confidence = run_rule_based(reading)
        ml_result = self.ml_engine.predict(reading)
        optimization = optimize_actuator_plan(reading, actuator, alert, scenario)

        literature = LITERATURE.get(scenario.value.lower(), LITERATURE["normal"])
        decision_comment = build_decision_comment(
            reading=reading,
            alert=alert,
            scenario=scenario,
            flags=flags,
            actuator=actuator,
            trend=ml_result.get("trend"),
            optimization=optimization,
        )
        user_view = build_user_view(
            reading=reading,
            alert=alert,
            scenario=scenario,
            actuator=actuator,
            optimization=optimization,
            comment=decision_comment,
        )

        return DecisionResult(
            timestamp=reading.timestamp,
            alert_level=alert,
            scenario=scenario,
            sensor_data={
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "water_level": reading.water_level,
                "N": reading.nitrogen,
                "P": reading.phosphorus,
                "K": reading.potassium,
            },
            actuator_command=actuator.to_dict(),
            rule_based_flags=flags,
            ml_cluster=ml_result.get("cluster"),
            ml_trend=ml_result.get("trend"),
            optimization=optimization,
            decision_comment=decision_comment,
            user_view=user_view,
            literature_refs=literature,
            user_action_required=(alert in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]),
            confidence_score=confidence,
        )


def load_dataset(path: str) -> tuple[pd.DataFrame, list[SensorReading]]:
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df = df.rename(columns={"tempreature": "temperature"})

    readings = []
    for _, row in df.iterrows():
        r = SensorReading(
            timestamp=str(row.get("date", datetime.now().isoformat())),
            temperature=float(row.get("temperature", 25.0)),
            humidity=float(row.get("humidity", 60.0)),
            water_level=float(row.get("water_level", 80.0)),
            nitrogen=int(row.get("N", 150)),
            phosphorus=int(row.get("P", 50)),
            potassium=int(row.get("K", 200)),
            plant_type="lettuce",
            fan_on=int(row.get("Fan_actuator_ON", 0)),
            watering_pump_on=int(row.get("Watering_plant_pump_ON", 0)),
            water_pump_on=int(row.get("Water_pump_actuator_ON", 0)),
        )
        readings.append(r)
    return df, readings
