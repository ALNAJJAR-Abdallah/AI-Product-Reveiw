# AI Product Review Intelligence System

## Project Overview

**AI Product Review Intelligence System** is a university AI project that analyzes product reviews and transforms them into business insights.

The system combines:

- A fine-tuned **DistilBERT** sentiment analysis model
- A **Streamlit** web application
- A **multi-agent architecture**
- **CrewAI** agent definitions
- **Gemini API** report generation
- **Amazon Reviews 2023** product-level review search
- JSON logging and centralized error handling
- A human-in-the-loop validation checkpoint

The detailed technical explanation, model training process, architecture justification, and screenshots are provided in the project report.

---

## Main Features

- Product review collection from multiple sources
- Sentiment analysis using a fine-tuned DistilBERT model
- Positive / negative / neutral review percentages
- Product strengths and common complaints extraction
- Human approval before final report generation
- Final report generation with Gemini API
- JSON logs displayed inside the Streamlit app
- Modular and merge-friendly project architecture

---

## Academic Requirements Covered

| Requirement | Status |
|---|---|
| Multi-agent architecture | Implemented |
| At least 2 specialist agents + 1 orchestrator | Implemented |
| Fine-tuned deep learning model | Implemented |
| Human-in-the-loop checkpoint | Implemented |
| Logging | Implemented |
| Error handling | Implemented |
| Streamlit app | Implemented |
| CrewAI | Implemented |
| Gemini API | Implemented |
| Product review analysis | Implemented |

---

## Team Responsibilities

### ML / Deep Learning Part

The ML part includes:

- Dataset preprocessing
- DistilBERT fine-tuning
- Model evaluation
- Confusion matrix
- Classification report
- Prediction API

Main file:

```text
src/sentiment_model.py
```

Prediction function:

```python
from src.sentiment_model import predict_sentiment

result = predict_sentiment("This product is amazing.")
print(result)
```

Expected output:

```python
{
    "label": "positive",
    "score": 0.95
}
```

### Application / Multi-Agent Part

The application part includes:

- Streamlit interface
- CrewAI agent definitions
- Orchestrator
- Collector Agent
- Sentiment Agent integration
- Insight Agent
- Report Agent
- JSON logging
- Error handling
- Human-in-the-loop approval
- Amazon Reviews 2023 product search
- Gemini report generation

---

## Global Architecture

```text
User
 |
 v
Streamlit App
 |
 v
Orchestrator
 |
 +--> Collector Agent
 |
 +--> Sentiment Agent
 |
 +--> Insight Agent
 |
 +--> Human Approval Checkpoint
 |
 +--> Report Agent
```

### Agents

- **Collector Agent**: collects product reviews from selected sources.
- **Sentiment Agent**: uses the fine-tuned DistilBERT model to classify reviews.
- **Insight Agent**: computes sentiment percentages, complaints, and strengths.
- **Report Agent**: generates the final report after human approval.
- **Orchestrator**: coordinates the full workflow.

---

## Review Sources

The app supports multiple review sources:

1. **Manual reviews**  
   The user can paste reviews directly in Streamlit.

2. **Demo CSV**  
   Local fallback dataset:

   ```text
   data/sample_reviews.csv
   ```

3. **Hugging Face lightweight reviews**  
   Useful for quick tests.

4. **Amazon Reviews 2023 product-level data**  
   Uses product metadata, `parent_asin`, and local SQLite indexes to retrieve product-level reviews.

---

## Amazon Reviews 2023 Indexing

Amazon Reviews 2023 is too large to load fully in memory.

To keep the app fast, the project uses a local SQLite index.

Generated database:

```text
data/amazon2023_product_index.db
```

This file is generated locally and is **not pushed to GitHub**.

### Build Product Index

```bash
python scripts/build_amazon2023_index.py --max-per-category 100000
```

### Build Reviews Index

For the demo, the following categories are enough:

```bash
python scripts/build_amazon2023_reviews_index.py --categories Cell_Phones_and_Accessories Clothing_Shoes_and_Jewelry --max-scan-per-category 1000000 --max-reviews-per-product 20
```

After that, Streamlit can retrieve reviews from SQLite instead of scanning remote Amazon files.

---

## Project Structure

```text
AI-Product-Review-Intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   └── sample_reviews.csv
│
├── logs/
│   └── .gitkeep
│
├── reports/
│   └── .gitkeep
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── evaluation_metrics.json
│
├── scripts/
│   ├── build_amazon2023_index.py
│   ├── build_amazon2023_reviews_index.py
│   └── save_results.py
│
├── src/
│   ├── sentiment_model.py
│   │
│   ├── app_agents/
│   │   └── crew_agents.py
│   │
│   ├── app_tools/
│   │   ├── amazon2023_collector.py
│   │   ├── hf_review_collector.py
│   │   ├── insight_tool.py
│   │   ├── product_index_search.py
│   │   ├── report_tool.py
│   │   ├── review_collector.py
│   │   └── sentiment_tool.py
│   │
│   ├── orchestration/
│   │   └── orchestrator.py
│   │
│   └── utils/
│       ├── errors.py
│       └── logger.py
```

---

## Model Setup

The trained DistilBERT model is not included in GitHub because it is too large.

Place the model folder here:

```text
model/sentiment_distilbert/
```

Expected structure:

```text
model/
└── sentiment_distilbert/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── ...
```

Test the model:

```bash
python -c "from src.sentiment_model import predict_sentiment; print(predict_sentiment('This product is amazing.'))"
```

Expected output:

```python
{'label': 'positive', 'score': 0.9962}
```

---

## Gemini API Setup

Create a `.env` file at the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
CREWAI_LLM_MODEL=gemini/gemini-2.5-flash
```

A safe template is provided:

```text
.env.example
```

Important:

```text
.env must never be pushed to GitHub.
```

If no Gemini API key is configured, the app still works using a fallback report generator.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ALNAJJAR-Abdallah/AI-Product-Reveiw.git
cd AI-Product-Reveiw
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Compatible versions used during development:

```text
datasets==3.6.0
huggingface_hub==0.36.2
transformers==4.53.3
tokenizers==0.21.4
```

---

## Run the Application

```bash
streamlit run app.py
```

In the Streamlit interface:

1. Choose a review source.
2. Enter a product name.
3. Click **Analyze Reviews**.
4. Review sentiment results.
5. Validate the human approval checkbox.
6. Click **Generate Final Report**.
7. View logs in the logs viewer.

---

## Recommended Demo Inputs

Recommended source:

```text
Amazon Reviews 2023 product-level
```

Good examples:

```text
iPhone case
RayBan black
screen protector
phone charger
```

These examples return enough reviews for screenshots and demonstrate the full workflow.

---

## Logs and Reports

Runtime logs:

```text
logs/agent_actions.jsonl
```

Generated reports:

```text
reports/
```

These files are generated during execution and are not pushed to GitHub.

---

## Git Ignore Rules

The following files and folders are ignored:

```text
.venv/
venv/
.env
model/
logs/*.jsonl
reports/*.md
data/amazon2023_product_index.db
__pycache__/
*.pyc
.ipynb_checkpoints/
.cache/
```

Reason:

- `.env` contains private API keys.
- `model/` contains large trained model files.
- SQLite indexes are generated locally.
- Logs and reports are runtime outputs.
- `.venv/` is a local environment folder.

---

## Current Status

```text
READY FOR SUBMISSION
```

Completed:

- Fine-tuned DistilBERT model integration
- Streamlit application
- Multi-agent architecture
- CrewAI agent definitions
- Orchestrator workflow
- Human-in-the-loop checkpoint
- JSON logging
- Error handling
- Gemini report generation
- Amazon Reviews 2023 product-level search
- Local SQLite product and reviews indexing
- Final README and requirements file

---

## Short Presentation Summary

The system starts in Streamlit, where the user enters a product name.  
The Orchestrator coordinates the agents. The Collector Agent retrieves reviews, the Sentiment Agent classifies each review using the fine-tuned DistilBERT model, the Insight Agent extracts business insights, and the Report Agent generates the final report after human approval.

This architecture satisfies the main requirements: fine-tuned deep learning model integration, multi-agent collaboration, orchestrator, human-in-the-loop checkpoint, logging, error handling, Streamlit interface, and report generation.
