"""
Smart Co-Pilot — FastAPI Gateway
"""

import json
import os
import base64
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
except Exception:  # pragma: no cover - optional dependency at runtime
    google = None
    GoogleAuthRequest = None

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from decision_engine import (
    HybridDecisionEngine,
    PLANT_PROFILES,
    SensorReading,
    THRESHOLDS,
    LITERATURE,
    evaluate_what_if,
    load_dataset,
)

engine = None
latest_reading: Optional[SensorReading] = None
latest_decision: Optional[dict] = None
active_plant_type = "lettuce"
latest_air_quality: Optional[dict] = None
demo_buffers: dict[str, list[SensorReading]] = {}
demo_cursor: dict[str, int] = {"heatwave": 0, "drought": 0}
demo_mixed_cursor = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, demo_buffers
    engine = HybridDecisionEngine()
    try:
        df1 = pd.read_excel("data/synthetic_heatwave_scenario.xlsx")
        df2 = pd.read_excel("data/synthetic_drought_scenario.xlsx")
        combined = pd.concat([df1, df2], ignore_index=True)
        engine.fit(combined)
        _, heatwave = load_dataset("data/synthetic_heatwave_scenario.xlsx")
        _, drought = load_dataset("data/synthetic_drought_scenario.xlsx")
        demo_buffers = {"heatwave": heatwave, "drought": drought}
        print("[Startup] ML modeli eğitildi.")
    except Exception as e:
        print(f"[Startup] Eğitim verisi yüklenemedi: {e} — Engine hazır ama ML pasif.")
    yield
    print("[Shutdown] Engine kapatıldı.")


app = FastAPI(
    title="Smart Co-Pilot Decision Engine",
    description="IoT tabanlı Akıllı Tarım Karar Destek Sistemi — Rule-Based + ML + Optimization",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class SensorPayload(BaseModel):
    timestamp: Optional[str] = None
    temperature: float = Field(ge=-20, le=70)
    humidity: float = Field(ge=0, le=100)
    water_level: float = Field(ge=0, le=100)
    nitrogen: int = Field(default=150, ge=0, le=500)
    phosphorus: int = Field(default=50, ge=0, le=500)
    potassium: int = Field(default=200, ge=0, le=500)
    plant_type: str = "lettuce"
    fan_on: int = Field(default=0, ge=0, le=1)
    watering_pump_on: int = Field(default=0, ge=0, le=1)
    water_pump_on: int = Field(default=0, ge=0, le=1)


class WhatIfPayload(BaseModel):
    fan: bool = False
    watering_pump: bool = False
    water_pump: bool = False


class PlantSelectPayload(BaseModel):
    plant_type: str


class ExplainPayload(BaseModel):
    language: str = "en"


class AskPayload(BaseModel):
    question: str
    language: str = "en"


def _to_sensor_reading(payload: SensorPayload) -> SensorReading:
    plant_type = payload.plant_type if payload.plant_type in PLANT_PROFILES else active_plant_type
    return SensorReading(
        timestamp=payload.timestamp or datetime.now().isoformat(),
        temperature=payload.temperature,
        humidity=payload.humidity,
        water_level=payload.water_level,
        nitrogen=payload.nitrogen,
        phosphorus=payload.phosphorus,
        potassium=payload.potassium,
        plant_type=plant_type,
        fan_on=payload.fan_on,
        watering_pump_on=payload.watering_pump_on,
        water_pump_on=payload.water_pump_on,
    )


def _decide_and_store(reading: SensorReading) -> dict:
    global latest_reading, latest_decision
    result = engine.decide(reading)
    latest_reading = reading
    latest_decision = json.loads(result.to_json())
    if latest_air_quality is not None:
        latest_decision["air_quality"] = latest_air_quality
    return latest_decision


def _copy_reading(base: SensorReading, **overrides) -> SensorReading:
    data = {
        "timestamp": base.timestamp,
        "temperature": base.temperature,
        "humidity": base.humidity,
        "water_level": base.water_level,
        "nitrogen": base.nitrogen,
        "phosphorus": base.phosphorus,
        "potassium": base.potassium,
        "plant_type": base.plant_type,
        "fan_on": base.fan_on,
        "watering_pump_on": base.watering_pump_on,
        "water_pump_on": base.water_pump_on,
    }
    data.update(overrides)
    return SensorReading(**data)


def _gemini_fallback_explanation(decision: dict, language: str = "en") -> str:
    sensor = decision.get("sensor_data", {})
    user_view = decision.get("user_view", {})
    actions = user_view.get("action_recommendations", [])
    top = actions[0] if actions else {}
    scenario = user_view.get("scenario_label", decision.get("scenario", "-"))
    status = user_view.get("status_label", decision.get("alert_level", "-"))
    plant = user_view.get("plant_label", "-")
    temp = sensor.get("temperature", "-")
    hum = sensor.get("humidity", "-")
    water = sensor.get("water_level", "-")

    if language.lower().startswith("tr"):
        action_title = top.get("title", "Düzeni koru")
        action_detail = top.get("detail", "Şu anki düzeni izlemek yeterli görünüyor.")
        condition_line = (
            "Bitki şu an dengede görünüyor."
            if str(status).lower() in ("iyi", "gayet iyi", "ok")
            else f"Bitki şu anda '{scenario}' durumuna yakın çalışıyor."
        )
        return (
            f"Ben olsam bunu şöyle okurum: {condition_line}\n\n"
            f"1. Şu anda ne görüyorum?\n"
            f"Sıcaklık {temp}, nem {hum} ve su seviyesi {water}. Bu tablo {plant} için sistemin dikkat ettiği ana resmi veriyor.\n\n"
            f"2. Bu kullanıcı için ne anlama geliyor?\n"
            f"Sistem bu yüzden durumu '{scenario}' olarak yorumluyor. Yani şu an bakılması gereken konu bu alan.\n\n"
            f"3. İlk yapılacak en doğru şey ne?\n"
            f"{action_title}. {action_detail}\n\n"
            f"Kısa sonuç: Önce bu adımı uygulamak en güvenli başlangıç olur."
        )

    action_title = top.get("title", "Keep the current setup")
    action_detail = top.get("detail", "The current setup looks stable for now.")
    return (
        f"This is how I would explain it: the plant currently looks close to '{scenario}'.\n\n"
        f"1. What am I seeing?\n"
        f"Temperature is {temp}, humidity is {hum}, and water level is {water}. These are the main signals behind the decision.\n\n"
        f"2. What does this mean for the user?\n"
        f"The system reads the current condition as '{scenario}', so this is the main area needing attention.\n\n"
        f"3. What should be done first?\n"
        f"{action_title}. {action_detail}\n\n"
        f"Short conclusion: Starting with this action is the safest first step."
    )


def _gemini_fallback_sections(decision: dict, language: str = "en") -> dict:
    sensor = decision.get("sensor_data", {})
    user_view = decision.get("user_view", {})
    actions = user_view.get("action_recommendations", [])
    top = actions[0] if actions else {}
    scenario = user_view.get("scenario_label", decision.get("scenario", "-"))
    status = user_view.get("status_label", decision.get("alert_level", "-"))
    plant = user_view.get("plant_label", "-")
    temp = sensor.get("temperature", "-")
    hum = sensor.get("humidity", "-")
    water = sensor.get("water_level", "-")
    action_title = top.get("title", "Düzeni koru")
    action_detail = top.get("detail", "Şu anki düzeni izlemek yeterli görünüyor.")

    if language.lower().startswith("tr"):
        headline = (
            f"{plant} şu anda genel olarak dengede."
            if str(status).lower() in ("iyi", "gayet iyi", "ok")
            else f"{plant} şu anda {scenario.lower()} tarafına kayıyor."
        )
        return {
            "headline": headline,
            "sections": [
                {
                    "label": "Benim kısa okumam",
                    "text": f"Sistem şu anda asıl dikkat edilmesi gereken konuyu '{scenario}' olarak görüyor.",
                },
                {
                    "label": "Neye bakarak bunu söylüyor?",
                    "text": f"Sıcaklık {temp}, nem {hum} ve su seviyesi {water}. Bu üç değer kararın temelini oluşturuyor.",
                },
                {
                    "label": "İlk yapılacak şey",
                    "text": f"{action_title}. {action_detail}",
                },
                {
                    "label": "Kısa tavsiye",
                    "text": "Önce bu adımı uygulayıp sonra sistemin yeniden nasıl tepki verdiğine bakmak en güvenli yaklaşım olur.",
                },
            ],
        }

    return {
        "headline": f"{plant} is currently leaning toward {scenario}.",
        "sections": [
            {
                "label": "My quick reading",
                "text": f"The system sees '{scenario}' as the main issue right now.",
            },
            {
                "label": "What is this based on?",
                "text": f"Temperature {temp}, humidity {hum}, and water level {water} are the main signals behind this decision.",
            },
            {
                "label": "What should happen first?",
                "text": f"{action_title}. {action_detail}",
            },
            {
                "label": "Short advice",
                "text": "Start with this step first, then watch how the system responds.",
            },
        ],
    }


def _build_gemini_prompt(decision: dict, language: str = "en") -> str:
    sensor = decision.get("sensor_data", {})
    user_view = decision.get("user_view", {})
    recommendations = user_view.get("action_recommendations", [])
    rec_text = "; ".join(
        f"{item.get('title', '-')}: {item.get('detail', '-')}"
        for item in recommendations[:3]
    ) or "-"
    lang_instruction = (
        "Write in Turkish."
        if language.lower().startswith("tr")
        else "Write in English."
    )
    return (
        "You are an agronomist explaining a smart farming dashboard to a person who knows almost nothing about agriculture. "
        "Do not change the decision. Explain only the existing backend decision in a calm, simple, reassuring, user-friendly way. "
        "Keep it short. Use plain words. Avoid jargon unless you immediately simplify it. "
        "Structure the answer in 4 compact parts: a short headline, what I see, what it means, what to do first. "
        "Then end with one very short concluding sentence. "
        "Do not repeat every raw metric mechanically. Prioritize only what matters most for the user. "
        "Sound like a calm agronomist, not like a machine log. Avoid hype, avoid unsupported claims. "
        f"{lang_instruction}\n\n"
        f"Alert level: {decision.get('alert_level', '-')}\n"
        f"Scenario: {decision.get('scenario', '-')}\n"
        f"Plant: {user_view.get('plant_label', '-')}\n"
        f"Temperature: {sensor.get('temperature', '-')}\n"
        f"Humidity: {sensor.get('humidity', '-')}\n"
        f"Water level: {sensor.get('water_level', '-')}\n"
        f"Recommended actions: {rec_text}\n"
        f"Decision summary: {user_view.get('summary', decision.get('decision_comment', '-'))}\n"
    )


def _build_gemini_question_prompt(decision: dict, question: str, language: str = "en") -> str:
    sensor = decision.get("sensor_data", {})
    user_view = decision.get("user_view", {})
    recommendations = user_view.get("action_recommendations", [])
    rec_text = "; ".join(
        f"{item.get('title', '-')}: {item.get('detail', '-')}"
        for item in recommendations[:3]
    ) or "-"
    lang_instruction = (
        "Write in Turkish."
        if language.lower().startswith("tr")
        else "Write in English."
    )
    return (
        "You are an agronomist answering a user's question about a smart farming dashboard. "
        "Answer only from the current backend decision context. "
        "Keep the answer short, simple, and reassuring. "
        "Assume the user knows very little technical detail. "
        "Use at most 4 short sentences. Avoid jargon. "
        f"{lang_instruction}\n\n"
        f"User question: {question}\n"
        f"Alert level: {decision.get('alert_level', '-')}\n"
        f"Scenario: {decision.get('scenario', '-')}\n"
        f"Plant: {user_view.get('plant_label', '-')}\n"
        f"Temperature: {sensor.get('temperature', '-')}\n"
        f"Humidity: {sensor.get('humidity', '-')}\n"
        f"Water level: {sensor.get('water_level', '-')}\n"
        f"Recommended actions: {rec_text}\n"
        f"Decision summary: {user_view.get('summary', decision.get('decision_comment', '-'))}\n"
    )


def _gemini_fallback_answer(decision: dict, question: str, language: str = "en") -> str:
    q = question.lower()
    user_view = decision.get("user_view", {})
    sensor = decision.get("sensor_data", {})
    top = (user_view.get("action_recommendations") or [{}])[0]
    scenario = user_view.get("scenario_label", decision.get("scenario", "-"))
    if language.lower().startswith("tr"):
        if "neden" in q:
            return f"Sistem şu anda ana konuyu '{scenario}' olarak görüyor. Bu yüzden ilk öneri olarak '{top.get('title', 'mevcut düzeni koru')}' öne çıkıyor."
        if "sulama" in q:
            return f"Sulama konusu önemli çünkü nem ve su dengesi bitkinin stres yaşayıp yaşamamasını doğrudan etkiliyor. Şu anki öneri bu dengeyi korumaya odaklanıyor."
        if "fan" in q:
            return f"Fan, sıcaklık ve hava akışını etkilemek için kullanılır. Ama her senaryoda iyi sonuç vermez; örneğin soğukta ortamı daha da soğutabilir."
        return (
            f"Kısa cevap: Sistem şu anda '{scenario}' durumunu yönetmeye çalışıyor. "
            f"Bu yüzden '{top.get('title', 'mevcut düzeni koru')}' adımı öncelikli görünüyor."
        )
    return (
        f"Short answer: the system is currently focused on '{scenario}'. "
        f"That is why '{top.get('title', 'keep the current setup')}' is shown as the first action."
    )


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "\n".join(texts).strip()


def _get_vertex_settings() -> dict:
    project = (
        os.getenv("VERTEX_AI_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )
    location = os.getenv("VERTEX_AI_LOCATION", "global")
    model = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash")
    enabled = bool(project)
    return {
        "enabled": enabled,
        "project": project,
        "location": location,
        "model": model,
    }


def _get_vertex_access_token() -> Optional[str]:
    if google is None or GoogleAuthRequest is None:
        return _get_vertex_access_token_from_service_account()
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(GoogleAuthRequest())
        return credentials.token
    except Exception:
        return _get_vertex_access_token_from_service_account()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _get_vertex_access_token_from_service_account() -> Optional[str]:
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        return None
    try:
        with open(creds_path, "r", encoding="utf-8") as f:
            sa = json.load(f)
        client_email = sa["client_email"]
        private_key = sa["private_key"]
        token_uri = sa.get("token_uri", "https://oauth2.googleapis.com/token")

        now = int(time.time())
        header = {"alg": "RS256", "typ": "JWT"}
        claim = {
            "iss": client_email,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": token_uri,
            "exp": now + 3600,
            "iat": now,
        }
        signing_input = (
            f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}."
            f"{_b64url(json.dumps(claim, separators=(',', ':')).encode())}"
        )

        with tempfile.NamedTemporaryFile("w", delete=False) as key_file:
            key_file.write(private_key)
            key_path = key_file.name
        try:
            proc = subprocess.run(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-sign",
                    key_path,
                ],
                input=signing_input.encode("utf-8"),
                capture_output=True,
                check=True,
            )
        finally:
            os.unlink(key_path)

        assertion = f"{signing_input}.{_b64url(proc.stdout)}"
        body = urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            }
        ).encode("utf-8")
        req = Request(
            token_uri,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("access_token")
    except Exception as e:
        print(f"[Vertex Gemini] Service-account token fallback basarisiz: {type(e).__name__}: {e}")
        return None


def _vertex_generate_content(prompt_text: str, model: str, project: str, location: str) -> str:
    token = _get_vertex_access_token()
    if not token:
        raise RuntimeError("Vertex access token alinamadi")

    base_url = (
        "https://aiplatform.googleapis.com/v1/"
        if location == "global"
        else f"https://{location}-aiplatform.googleapis.com/v1/"
    )
    url = (
        f"{base_url}projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}],
            }
        ]
    }
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _extract_gemini_text(payload)


def _generate_gemini_explanation(decision: dict, language: str = "en") -> dict:
    vertex = _get_vertex_settings()
    if vertex["enabled"]:
        try:
            explanation = _vertex_generate_content(
                _build_gemini_prompt(decision, language),
                model=vertex["model"],
                project=vertex["project"],
                location=vertex["location"],
            )
            if not explanation:
                raise ValueError("Vertex Gemini bos aciklama dondurdu")
            return {
                "source": "vertex-gemini",
                "model": vertex["model"],
                "generated_at": datetime.now().isoformat(),
                "explanation": explanation,
                "note": None,
            }
        except Exception as e:
            print(f"[Vertex Gemini] Explanation fallback'a dustu: {type(e).__name__}: {e}")
            fallback = _gemini_fallback_sections(decision, language)
            return {
                "source": "local-fallback",
                "model": vertex["model"],
                "generated_at": datetime.now().isoformat(),
                "explanation": _gemini_fallback_explanation(decision, language),
                "headline": fallback["headline"],
                "sections": fallback["sections"],
                "note": "Vertex Gemini erisimi basarisiz oldugu icin yerel ozet gosterildi.",
            }

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        fallback = _gemini_fallback_sections(decision, language)
        return {
            "source": "local-fallback",
            "model": None,
            "generated_at": datetime.now().isoformat(),
            "explanation": _gemini_fallback_explanation(decision, language),
            "headline": fallback["headline"],
            "sections": fallback["sections"],
            "note": "Gemini API key bulunamadi; yerel ozet gosterildi.",
        }

    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_gemini_prompt(decision, language),
                    }
                ]
            }
        ]
    }
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        explanation = _extract_gemini_text(payload)
        if not explanation:
            raise ValueError("Gemini bos aciklama dondurdu")
        return {
            "source": "gemini",
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "explanation": explanation,
            "note": None,
        }
    except Exception:
        fallback = _gemini_fallback_sections(decision, language)
        return {
            "source": "local-fallback",
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "explanation": _gemini_fallback_explanation(decision, language),
            "headline": fallback["headline"],
            "sections": fallback["sections"],
            "note": "Gemini erisimi basarisiz oldugu icin yerel ozet gosterildi.",
        }


def _generate_gemini_answer(decision: dict, question: str, language: str = "en") -> dict:
    vertex = _get_vertex_settings()
    if vertex["enabled"]:
        try:
            answer = _vertex_generate_content(
                _build_gemini_question_prompt(decision, question, language),
                model=vertex["model"],
                project=vertex["project"],
                location=vertex["location"],
            )
            if not answer:
                raise ValueError("Vertex Gemini bos yanit dondurdu")
            return {
                "source": "vertex-gemini",
                "model": vertex["model"],
                "generated_at": datetime.now().isoformat(),
                "answer": answer,
                "note": None,
            }
        except Exception as e:
            print(f"[Vertex Gemini] Question-answer fallback'a dustu: {type(e).__name__}: {e}")
            return {
                "source": "local-fallback",
                "model": vertex["model"],
                "generated_at": datetime.now().isoformat(),
                "answer": _gemini_fallback_answer(decision, question, language),
                "note": "Vertex Gemini erisimi basarisiz oldugu icin ornek yanit gosterildi.",
            }

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "source": "local-fallback",
            "model": None,
            "generated_at": datetime.now().isoformat(),
            "answer": _gemini_fallback_answer(decision, question, language),
            "note": "Gemini API key bulunamadi; ornek yanit gosterildi.",
        }

    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_gemini_question_prompt(decision, question, language),
                    }
                ]
            }
        ]
    }
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        answer = _extract_gemini_text(payload)
        if not answer:
            raise ValueError("Gemini bos yanit dondurdu")
        return {
            "source": "gemini",
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "answer": answer,
            "note": None,
        }
    except Exception:
        return {
            "source": "local-fallback",
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "answer": _gemini_fallback_answer(decision, question, language),
            "note": "Gemini erisimi basarisiz oldugu icin ornek yanit gosterildi.",
        }


@app.get("/")
def dashboard_home():
    return FileResponse(static_dir / "dashboard.html")


@app.get("/health")
def health():
    vertex = _get_vertex_settings()
    return {
        "status": "ok",
        "ml_fitted": engine.ml_engine.fitted if engine else False,
        "decision_mode": "rule_based+ml+optimization",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "vertex_gemini_configured": vertex["enabled"],
        "vertex_location": vertex["location"] if vertex["enabled"] else None,
        "vertex_model": vertex["model"] if vertex["enabled"] else None,
        "has_current_data": latest_decision is not None,
        "has_air_quality": latest_air_quality is not None,
        "active_plant_type": active_plant_type,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/thresholds")
def get_thresholds():
    return {"thresholds": THRESHOLDS, "literature": LITERATURE}


@app.get("/plants")
def list_plants():
    return {
        "active_plant_type": active_plant_type,
        "plants": [
            {"id": k, "label": v["label"]}
            for k, v in PLANT_PROFILES.items()
        ],
    }


@app.post("/plants/select")
def select_plant(payload: PlantSelectPayload):
    global active_plant_type, latest_reading, latest_decision
    if payload.plant_type not in PLANT_PROFILES:
        raise HTTPException(400, f"Geçersiz bitki türü: {payload.plant_type}")
    active_plant_type = payload.plant_type
    if latest_reading is not None:
        latest_reading.plant_type = active_plant_type
        latest_decision = _decide_and_store(latest_reading)
    return {"status": "ok", "active_plant_type": active_plant_type}


@app.post("/ingest")
def ingest_sensor_data(payload: SensorPayload):
    """
    Sensör katmanından gelen otomatik veriyi işleyip güncel kararı saklar.
    """
    if not engine:
        raise HTTPException(503, "Engine başlatılmamış")

    reading = _to_sensor_reading(payload)
    decided = _decide_and_store(reading)
    return {
        "status": "ok",
        "timestamp": decided["timestamp"],
        "alert_level": decided["alert_level"],
    }


@app.get("/current")
def get_current_state():
    if latest_decision is None:
        raise HTTPException(404, "Henüz sensör verisi gelmedi")
    return latest_decision


@app.post("/demo/next")
def demo_next(scenario: str = "heatwave"):
    """
    Demo amaçlı bir sonraki sentetik sensör kaydını ingest eder.
    """
    global demo_mixed_cursor
    valid = list(demo_buffers.keys()) + ["mixed", "normal", "cold_stress", "hardware_fault"]
    if scenario not in valid:
        raise HTTPException(400, f"Geçersiz senaryo. Seçenekler: {valid}")

    if scenario == "heatwave":
        rows = demo_buffers.get(scenario, [])
        if not rows:
            raise HTTPException(404, "Demo veri bulunamadı")
        idx = demo_cursor[scenario] % len(rows)
        demo_cursor[scenario] += 1
        reading = _copy_reading(rows[idx], plant_type=active_plant_type)
    else:
        heat = demo_buffers.get("heatwave", [])
        drt = demo_buffers.get("drought", [])
        if not heat or not drt:
            raise HTTPException(404, "Karışık demo için temel veri bulunamadı")

        if scenario == "mixed":
            pattern = ["normal", "heatwave", "drought", "cold_stress", "hardware_fault"]
            scenario = pattern[demo_mixed_cursor % len(pattern)]
            demo_mixed_cursor += 1

        if scenario == "normal":
            base = heat[demo_cursor["heatwave"] % len(heat)]
            demo_cursor["heatwave"] += 1
            reading = _copy_reading(
                base,
                temperature=24.0,
                humidity=62.0,
                water_level=68.0,
                plant_type=active_plant_type,
                fan_on=0,
                watering_pump_on=0,
                water_pump_on=0,
            )
        elif scenario == "drought":
            base = drt[demo_cursor["drought"] % len(drt)]
            demo_cursor["drought"] += 1
            # Kuraklık: düşük nem + azalan su seviyesi (soğuk stresten belirgin farklı)
            reading = _copy_reading(
                base,
                temperature=34.5,
                humidity=21.0,
                water_level=26.0,
                plant_type=active_plant_type,
                fan_on=0,
                watering_pump_on=0,
                water_pump_on=0,
            )
        elif scenario == "cold_stress":
            base = heat[demo_cursor["heatwave"] % len(heat)]
            demo_cursor["heatwave"] += 1
            # Soğuk stres: düşük sıcaklık, nem/su görece dengeli
            reading = _copy_reading(
                base,
                temperature=6.8,
                humidity=63.0,
                water_level=61.0,
                plant_type=active_plant_type,
                fan_on=0,
                watering_pump_on=0,
                water_pump_on=0,
            )
        else:  # hardware_fault
            base = drt[demo_cursor["drought"] % len(drt)]
            demo_cursor["drought"] += 1
            reading = _copy_reading(
                base,
                humidity=28.0,
                water_level=46.0,
                plant_type=active_plant_type,
                watering_pump_on=1,
            )

    decided = _decide_and_store(reading)
    return {
        "status": "ok",
        "scenario": scenario,
        "cursor": demo_cursor.get(scenario, demo_mixed_cursor),
        "timestamp": decided["timestamp"],
        "alert_level": decided["alert_level"],
        "top_recommendation": (
            decided.get("user_view", {})
            .get("action_recommendations", [{}])[0]
            .get("title", "Öneri üretilemedi")
        ),
    }


@app.get("/air-quality/openaq")
def get_air_quality_openaq(city: str = "Istanbul", country: str = "TR"):
    """
    OpenAQ verisini çekmeyi dener; erişim yoksa kontrollü demo verisi döner.
    """
    global latest_air_quality, latest_decision
    try:
        # OpenAQ v3 uç noktası (anahtarsız erişimde sınırlı olabilir)
        url = (
            "https://api.openaq.org/v3/latest?"
            f"city={quote(city)}&country={quote(country)}&limit=5"
        )
        with urlopen(url, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        latest_air_quality = {
            "source": "OpenAQ",
            "city": city,
            "country": country,
            "fetched_at": datetime.now().isoformat(),
            "raw": payload,
        }
    except Exception:
        # Ağ/anahtar problemi durumunda demo fallback
        latest_air_quality = {
            "source": "demo-fallback",
            "city": city,
            "country": country,
            "fetched_at": datetime.now().isoformat(),
            "summary": {
                "pm25": 18.4,
                "pm10": 31.7,
                "o3": 52.1,
                "no2": 14.6,
            },
            "note": "OpenAQ erişimi başarısız olduğu için demo veri kullanıldı.",
        }
    if latest_decision is not None:
        latest_decision["air_quality"] = latest_air_quality
    return latest_air_quality


@app.post("/decide")
def decide_fast(payload: SensorPayload):
    """
    Geriye dönük uyumluluk için manuel değerlendirme endpoint'i.
    """
    if not engine:
        raise HTTPException(503, "Engine başlatılmamış")

    reading = _to_sensor_reading(payload)
    return _decide_and_store(reading)


@app.post("/what-if")
def what_if(payload: WhatIfPayload):
    if latest_reading is None:
        raise HTTPException(404, "What-if için önce sensör verisi gerekli")

    return evaluate_what_if(
        reading=latest_reading,
        fan=payload.fan,
        watering_pump=payload.watering_pump,
        water_pump=payload.water_pump,
    )


@app.post("/ai/explain")
def ai_explain(payload: ExplainPayload):
    if latest_decision is None:
        raise HTTPException(404, "Gemini aciklamasi icin once sensor verisi gerekli")
    return _generate_gemini_explanation(latest_decision, payload.language)


@app.post("/ai/ask")
def ai_ask(payload: AskPayload):
    if latest_decision is None:
        raise HTTPException(404, "AI soru-cevap icin once sensor verisi gerekli")
    if not payload.question.strip():
        raise HTTPException(400, "Soru bos olamaz")
    return _generate_gemini_answer(latest_decision, payload.question.strip(), payload.language)


@app.post("/simulate")
def simulate_scenario(scenario: str = "heatwave", rows: int = 10):
    global latest_reading, latest_decision

    paths = {
        "heatwave": "data/synthetic_heatwave_scenario.xlsx",
        "drought": "data/synthetic_drought_scenario.xlsx",
    }
    if scenario not in paths:
        raise HTTPException(400, f"Geçersiz senaryo. Seçenekler: {list(paths.keys())}")

    try:
        _, readings = load_dataset(paths[scenario])
    except FileNotFoundError:
        raise HTTPException(404, "Veri dosyası bulunamadı")

    results = []
    sim_engine = HybridDecisionEngine()
    try:
        df1 = pd.read_excel("data/synthetic_heatwave_scenario.xlsx")
        df2 = pd.read_excel("data/synthetic_drought_scenario.xlsx")
        sim_engine.fit(pd.concat([df1, df2], ignore_index=True))
    except Exception:
        pass

    for reading in readings[: min(rows, len(readings))]:
        result = sim_engine.decide(reading)
        parsed = json.loads(result.to_json())
        results.append(
            {
                "timestamp": parsed["timestamp"],
                "alert_level": parsed["alert_level"],
                "scenario": parsed["scenario"],
                "flags": parsed["rule_based_flags"],
                "actuator": parsed["actuator_command"],
                "optimization": parsed["optimization"],
            }
        )

    if results:
        latest_reading = readings[min(rows, len(readings)) - 1]
        latest_decision = json.loads(sim_engine.decide(latest_reading).to_json())

    return {"scenario": scenario, "total_rows": len(results), "results": results}


# uvicorn api:app --reload --port 8000
