# Switch Protocol AI Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)

The **Switch Protocol AI Engine** is the intelligence layer powering the Switch Social Market. It provides real-time analysis of product listings, seller reputation, and market trends to ensure safe and transparent transactions on the BNB Chain.

## 🧠 Core Capabilities

### 1. Market Value Analysis
- **Real-time Price Comparison**: Aggregates data from major marketplaces to determine fair market value ranges.
- **Trend Detection**: Identifies price fluctuations and seasonal trends for specific categories.

### 2. Trust & Safety Scoring
- **Seller Reputation Analysis**: Analyzes on-chain transaction history and off-chain social signals to calculate a dynamic trust score.
- **Fraud Detection**: Uses NLP to scan listing descriptions for red flags (e.g., "cash only", "no meetup").

### 3. Smart Checklists
- **Context-Aware Safety Tips**: Generates category-specific inspection checklists (e.g., "Check battery cycle count" for laptops vs. "Check stitching" for luxury bags).

## 🚀 API Endpoints

### `POST /analyze/product`
Analyzes a product listing and returns a comprehensive summary, price analysis, and safety checklist.

**Request:**
```json
{
  "title": "Nintendo Switch OLED",
  "price": 260,
  "description": "Like new condition...",
  "category": "Gaming"
}
```

**Response:**
```json
{
  "summary": "This item is listed 10% below market average...",
  "trust_score": 0.95,
  "checklist": ["Check screen for scratches", "Verify Joy-Con drift"]
}
```

## 🛠 Tech Stack

- **Framework**: FastAPI
- **ML/NLP**: LangChain, OpenAI GPT-4 (or Local LLM)
- **Data**: Pandas, NumPy
- **Deployment**: Docker, AWS Lambda

## 📦 Installation

```bash
git clone https://github.com/SWPL-Switch-Protocol/ai-engine.git
cd ai-engine
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🔗 Integration

This engine is currently integrated with the [Switch Frontend](https://github.com/SWPL-Switch-Protocol/frontend-web).
