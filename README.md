# 🤖 E-Commerce Multilingual Smart Support Router & Escalation Hub

An enterprise-grade, hybrid NLP gateway combining a **fine-tuned DistilBERT intent classifier**, deterministic business orchestration, and automated **multilingual translation layers** (`Helsinki-NLP` + `SentencePiece`) designed to satisfy international compliance and routing architectures.

🚀 **[Live Interactive Web Demo on Hugging Face Spaces](YOUR_HUGGINGFACE_SPACE_URL_HERE)**

---

## 📊 Visual System Architecture

The gateway handles incoming customer tickets asynchronously across language boundaries, deciding whether to serve automated deterministic templated responses or escalate to a human reviewer based on mathematical model calibration.

```text
                  [ User Query (EN or FR) ]
                             │
                             ▼
                [ Language Detection Layer ]
                             │
              ┌──────────────┴──────────────┐
       Detected: English             Detected: French
              │                             │
              │                             ▼
              │                  [ MarianMT Translation ]
              │                        (FR ──> EN)
              ▼                             │
     ┌──────────────────────────────────────┴┐
     │   DistilBERT Intent Classification    │
     └───────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
       Confidence ≥ 72%?             Confidence < 72%?
              │                             │
              ▼                             ▼
     [ Static Response ]           [ Fallback GenAI Draft ]
    (Deterministic Match)         (Human-in-the-Loop Flagged)
              │                             │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
       Original Query French?        Original Query English?
              │                             │
              ▼                             ▼
   [ MarianMT Translation ]          [ Final Response ]
         (EN ──> FR)                        │
              │                             │
              ▼                             ▼
       [ Final Response ]            [ Return Payload ]

---

## 🛠️ Core Engineering Features

* **Fine-Tuned Intent Core:** Fine-tuned an open-source **DistilBERT** sequence classifier, hitting an evaluation accuracy score of **96.8%** across critical operational e-commerce intents (`WHERE_IS_MY_ORDER`, `REFUND_REQUEST`, `CANCEL_ORDER`, `PRODUCT_FEEDBACK`).
* **Multilingual Localization Middleware:** Implemented an automated localization layer using **MarianMT Architecture** via Google's **SentencePiece** tokenization. This allows an English-trained classifier core to gracefully read, process, and accurately reply to international French client text.
* **FastAPI Calibration Routing Engine:** Configured an explicit **Confidence Score Threshold ($72\%$)** to orchestrate compliance workflows. Safe queries are automatically dispatched, while low-confidence queries or edge cases are flagged with automated **Human-In-The-Loop** alert states.
* **Dual-Pane Telemetry UI:** Wrapped the backend routing architecture inside a customized **Gradio Web Dashboard** that splits user-facing customer simulation from administrative agent telemetry.

---

## 📦 Project Architecture & Layout

```text
├── app/
│   ├── main.py        # FastAPI Backend Core, Translation Tensors & Thresholding
│   └── app.py         # Gradio Presentation Interface Layout & HTML Alert Renderers
├── requirements.txt   # Core Dependencies (transformers, torch, langdetect, etc.)
└── .gitignore         # Pycache, Virtual Environment & Token Exclusions
