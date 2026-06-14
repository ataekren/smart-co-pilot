# Smart Co-Pilot — Decision Engine

## Kurulum

```bash
pip install fastapi uvicorn scikit-learn pandas openpyxl google-auth requests
```

## Proje Yapısı

```
decision_engine/
├── decision_engine.py   ← Karar motoru (Rule-Based + ML + XAI)
├── api.py               ← FastAPI gateway
├── data/
│   ├── synthetic_heatwave_scenario.xlsx
│   └── synthetic_drought_scenario.xlsx
└── README.md
```

## Çalıştırma

```bash
# 1. Veri dosyalarını data/ klasörüne koy
# 2. Sunucuyu başlat
uvicorn api:app --reload --port 8000
```

## Endpoint'ler

| Method | Endpoint        | Açıklama                            | Süre     |
|--------|----------------|-------------------------------------|----------|
| GET    | /health        | Servis ve model durumu              | < 10ms   |
| GET    | /current       | Son sensör verisine göre anlık durum| < 10ms   |
| GET    | /thresholds    | Bitki eşik değerleri + literatür    | < 10ms   |
| POST   | /ingest        | Sensör verisini sisteme işler       | < 50ms   |
| POST   | /decide        | Manuel test amaçlı hızlı karar      | < 50ms   |
| POST   | /what-if       | Ayar değiştirirsem ne olur tahmini  | < 50ms   |
| POST   | /ai/explain    | Gemini destekli açıklama üretir     | ~1-3 sn  |
| POST   | /demo/next     | Demo sensör satırı içeri akıtır      | < 50ms   |
| GET    | /air-quality/openaq | OpenAQ hava kalitesi çekmeyi dener | ~1 sn |
| POST   | /simulate      | Dataset simülasyonu                 | < 200ms  |

## Gemini Açıklama Katmanı

- Ana karar motoru `rule-based + K-Means + optimization` olarak kalır.
- Gemini sadece mevcut karar için doğal dil açıklaması üretir.
- Vertex AI bilgileri tanımlıysa sistem Gemini'ye Vertex üzerinden gider.
- `GEMINI_API_KEY` veya `GOOGLE_API_KEY` tanımlıysa resmi Gemini API çağrılır.
- Anahtar yoksa veya erişim başarısızsa sistem yerel fallback özet gösterir.

## Vertex AI ile Kullanım

Vertex AI tercih edilecekse ortam değişkenleri:

```bash
export VERTEX_AI_PROJECT="your-gcp-project-id"
export VERTEX_AI_LOCATION="global"
export VERTEX_GEMINI_MODEL="gemini-2.5-flash"
```

Kimlik doğrulama için Application Default Credentials kullanılır. Örnek:

```bash
gcloud auth application-default login
```

Bu durumda `/ai/explain` ve `/ai/ask` endpoint'leri Gemini yanıtlarını Vertex AI üzerinden üretir.

## Örnek İstek (Next.js / cURL)

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"temperature": 41.5, "humidity": 22.0, "water_level": 75.0}'
```

## Örnek Yanıt

```json
{
  "timestamp": "2024-02-02T20:40:00",
  "alert_level": "CRITICAL",
  "scenario": "HEATWAVE",
  "actuator_command": {"fan": true, "watering_pump": false, "water_pump": false},
  "rule_based_flags": ["CRITICAL: Sıcaklık 41.5°C — Kritik ısı stresi"],
  "ml_cluster": 1,
  "ml_trend": {"temperature": 56.7, "humidity": 24.9, "water_level": 74.2},
  "user_action_required": true,
  "confidence_score": 1.0
}
```

## Mimari Karar Akışı

```
SensorReading
     │
     ▼
Phase 1: Rule-Based Engine    ← Deterministik, %100 güven, < 1ms
     │ (alert, scenario, flags, actuator)
     ▼
Phase 2: ML Engine            ← K-Means cluster + Trend regresyon
     │ (cluster_id, trend_30min)
     ▼
Phase 3: Optimization Layer   ← Objective function minimizasyonu
     │ (recommended_command, objective_score)
     ▼
DecisionResult JSON           ← FastAPI → Next.js Dashboard
```

## Donanım Entegrasyonu (ESP32/Raspberry Pi)

Gerçek donanım geldiğinde sadece veri kaynağını değiştirin:

```python
# Simülasyon (şu an)
payload = load_from_excel("data/heatwave.xlsx")

# Gerçek donanım (ileride)
payload = mqtt_client.subscribe("farm/sensors")
```
API endpoint'leri aynı kalır.
