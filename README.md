# AI Product Review Intelligence System

## 1. Project Overview

**AI Product Review Intelligence System** is a university multi-agent AI project that analyzes product reviews and transforms them into business insights.

The system combines:

- A fine-tuned **DistilBERT** sentiment analysis model.
- A modular **multi-agent architecture**.
- A **Streamlit** user interface.
- **CrewAI** agent organization.
- **Gemini API** report generation.
- A human-in-the-loop validation checkpoint.
- JSON logging and centralized error handling.

The goal is to allow a user to enter a product name, collect reviews, classify review sentiment, extract business insights, approve the analysis, and generate a final product intelligence report.

---

## 2. Academic Requirements Covered

The project satisfies the required constraints:

| Requirement | Status |
|---|---|
| Multi-agent architecture | Implemented |
| At least 2 specialist agents + 1 orchestrator | Implemented |
| Fine-tuned deep learning model | Implemented with DistilBERT |
| Human-in-the-loop checkpoint | Implemented before report generation |
| Logging | Implemented with JSON logs |
| Error handling | Implemented with custom exceptions |
| Streamlit app | Implemented |
| CrewAI | Agents defined and integrated |
| Gemini API | Used for final report generation |
| Product review analysis system | Implemented |

---

## 3. Team Responsibilities

### Person 1 — ML / Deep Learning Part

Person 1 was responsible for the full machine learning workflow:

- Dataset selection.
- Dataset preprocessing.
- Label encoding.
- Train / validation split.
- Tokenization.
- DistilBERT fine-tuning.
- Model evaluation.
- Confusion matrix generation.
- Classification report generation.
- Model saving.
- Creation of the prediction API.
- ML GitHub branch preparation.

### Person 2 — Application / Multi-Agent Part

Person 2 was responsible for the full application and integration layer:

- Streamlit interface.
- Multi-agent architecture.
- CrewAI agent definitions.
- Orchestrator workflow.
- Collector Agent.
- Sentiment Agent integration.
- Insight Agent.
- Report Agent.
- JSON logging.
- Error handling.
- Human-in-the-loop approval.
- Gemini report generation.
- GitHub integration.

---

## 4. Global Architecture

The system uses one orchestrator and four specialist agents.

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

The architecture is modular and easy to explain:

- The **Collector Agent** collects product reviews.
- The **Sentiment Agent** analyzes review sentiment using the fine-tuned DistilBERT model.
- The **Insight Agent** transforms sentiment predictions into business insights.
- The **Report Agent** generates the final product intelligence report.
- The **Orchestrator** coordinates the complete workflow.

---

## 5. Final Folder Structure

```text
AI-Product-Review-Intelligence/
│
├── app.py
│
├── data/
│   └── sample_reviews.csv
│
├── logs/
│   └── agent_actions.jsonl
│
├── model/
│   └── sentiment_distilbert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── ...
│
├── notebooks/
│   └── model_training_notebook.ipynb
│
├── reports/
│   └── generated_reports.md
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
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 6. Machine Learning Workflow

### 6.1 Dataset

The model was trained using:

```text
amazon_polarity
```

Reason for using this dataset:

- It is based on product reviews.
- It is compatible with HuggingFace.
- It contains positive and negative review labels.
- It is suitable for fast training during a 24-hour project.
- It is balanced enough for sentiment classification.

Original dataset fields:

```text
label
title
content
```

The `title` and `content` fields were merged into one field:

```text
review_text
```

Original labels:

| Original label | Meaning |
|---|---|
| 0 | negative |
| 1 | positive |

Final labels:

| Sentiment | Label |
|---|---|
| negative | 0 |
| positive | 1 |

---

### 6.2 Dataset Reduction

The original dataset was very large.

To respect the 24-hour deadline, the team selected:

```text
20,000 reviews
```

Final split:

| Split | Size |
|---|---|
| Training | 16,000 |
| Validation | 4,000 |

---

### 6.3 Model Choice

The selected model was:

```text
distilbert-base-uncased
```

Reason for choosing DistilBERT:

- Lighter than BERT.
- Faster to train.
- Smaller memory footprint.
- Strong NLP performance.
- Suitable for Google Colab.
- Suitable for a 24-hour academic project.

---

### 6.4 Tokenization

The tokenizer used was:

```python
DistilBertTokenizerFast
```

Tokenization parameters:

```python
truncation=True
padding="max_length"
max_length=128
```

Reason for `max_length=128`:

- Product reviews are usually short.
- It reduces memory usage.
- It speeds up training.
- It preserves enough context for sentiment analysis.

---

### 6.5 Fine-Tuning Configuration

The model was fine-tuned with HuggingFace Trainer.

Main training parameters:

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Weight decay | 0.01 |
| Evaluation strategy | Epoch |

Reason for only 2 epochs:

- The project deadline was short.
- DistilBERT was already pretrained.
- 2 epochs were enough to achieve strong results.
- It reduced the risk of overfitting.

---

### 6.6 Model Performance

Final validation results:

```text
Validation accuracy: 92.98%
Evaluation loss: 0.250581
```

Classification report summary:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.93 | 0.93 | 0.93 |
| Positive | 0.93 | 0.93 | 0.93 |

The model achieved balanced performance on both positive and negative reviews.

---

### 6.7 Saved ML Artifacts

The following artifacts were generated:

```text
results/classification_report.txt
results/confusion_matrix.png
results/evaluation_metrics.json
logs/training.log
```

The trained model and tokenizer were saved locally in:

```text
model/sentiment_distilbert/
```

Expected model folder:

```text
model/
└── sentiment_distilbert/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── training_args.bin
    ├── vocab.txt
    └── ...
```

---

## 7. Sentiment Prediction API

The ML part exposes a simple prediction function:

```python
from src.sentiment_model import predict_sentiment
```

Example usage:

```python
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

This function is used by:

- Streamlit.
- Sentiment Agent.
- CrewAI workflow.
- Orchestrator.

---

## 8. Important Model Note

The trained model folder is **not pushed to GitHub** because the model file is too large.

GitHub rejected:

```text
model.safetensors
```

Reason:

```text
255 MB > GitHub 100 MB file limit
```

Therefore, the model must be shared separately and placed manually here:

```text
AI-Product-Review-Intelligence/model/sentiment_distilbert/
```

After placing the model folder, test it with:

```bash
python -c "from src.sentiment_model import predict_sentiment; print(predict_sentiment('This product is amazing.'))"
```

Expected output:

```python
{'label': 'positive', 'score': 0.9962}
```

---

## 9. Multi-Agent Architecture

### 9.1 Orchestrator

Main file:

```text
src/orchestration/orchestrator.py
```

The orchestrator controls the complete workflow.

Workflow:

```text
1. Collect reviews
2. Analyze sentiment
3. Generate insights
4. Wait for human approval
5. Generate final report
```

The orchestrator ensures that the report cannot be generated until the user validates the analysis.

---

### 9.2 Collector Agent

Main files:

```text
src/app_tools/review_collector.py
src/app_tools/hf_review_collector.py
src/app_tools/amazon2023_collector.py
src/app_tools/product_index_search.py
```

The Collector Agent collects reviews from multiple possible sources.

Available review sources:

```text
1. Amazon Reviews 2023 product-level data
2. Hugging Face lightweight reviews
3. Local demo CSV
4. Manual user input
```

Example output:

```python
[
    {
        "review_id": 1,
        "product": "iPhone 12",
        "text": "The phone is fast and the screen quality is excellent.",
        "source": "amazon_reviews_2023_native"
    }
]
```

---

### 9.3 Sentiment Agent

Main file:

```text
src/app_tools/sentiment_tool.py
```

The Sentiment Agent calls:

```python
from src.sentiment_model import predict_sentiment
```

It enriches each review with:

- Sentiment label.
- Confidence score.
- Model source.

Example enriched review:

```python
{
    "text": "This product is amazing.",
    "sentiment_label": "positive",
    "sentiment_score": 0.9962,
    "model_used": "teammate_distilbert"
}
```

If the local fine-tuned model folder is missing, the system can use a temporary Transformers sentiment pipeline as fallback so the demo remains stable.

---

### 9.4 Insight Agent

Main file:

```text
src/app_tools/insight_tool.py
```

The Insight Agent calculates:

- Positive percentage.
- Negative percentage.
- Neutral percentage.
- Common complaints.
- Product strengths.

Example output:

```python
{
    "total_reviews": 10,
    "sentiment_counts": {
        "positive": 6,
        "negative": 3,
        "neutral": 1
    },
    "sentiment_percentages": {
        "positive": 60.0,
        "negative": 30.0,
        "neutral": 10.0
    },
    "common_complaints": ["battery mentioned 2 time(s)"],
    "strengths": ["camera mentioned 3 time(s)"]
}
```

---

### 9.5 Report Agent

Main file:

```text
src/app_tools/report_tool.py
```

The Report Agent generates the final product intelligence report.

It uses the Gemini API when configured.

If Gemini is not configured, a fallback Markdown report is generated so the demo does not crash.

The final report includes:

```text
1. Executive summary
2. Sentiment distribution
3. Main customer complaints
4. Product strengths
5. Business recommendations
6. Risks and limitations
7. Conclusion
```

---

## 10. Streamlit Application

Main file:

```text
app.py
```

Run the app with:

```bash
streamlit run app.py
```

The Streamlit app includes:

- Product name input.
- Review source selector.
- Optional manual review input.
- Sentiment metrics.
- Sentiment chart.
- Analyzed reviews table.
- Human approval checkbox.
- Final report generation button.
- JSON logs viewer.

Recommended demo source:

```text
Amazon Reviews 2023 product-level
```

Example product searches:

```text
iPhone 12
iPhone 12 charger
RayBan black
headphones
coffee machine
```

---

## 11. Review Sources

### 11.1 Manual Reviews

The user can paste reviews directly in Streamlit.

Manual reviews have priority over other sources.

This is useful for testing the complete workflow with controlled examples.

Example manual reviews:

```text
The iPhone 12 is fast, smooth, and the screen quality is excellent.
Battery life is not great and I need to charge it before the end of the day.
The camera quality is very good, especially for photos during the day.
The phone feels premium and the design is really clean.
```

---

### 11.2 Demo CSV

The demo CSV is used as a stable fallback.

File:

```text
data/sample_reviews.csv
```

This allows the application to work even if external datasets are unavailable.

---

### 11.3 Hugging Face Lightweight Reviews

This source uses a lightweight Amazon review dataset.

It is useful for quick tests.

Limitation:

```text
Product matching is keyword-based.
```

---

### 11.4 Amazon Reviews 2023 Product-Level Data

This is the most advanced review source.

It uses:

- Product metadata.
- Product title.
- Category.
- `parent_asin`.
- Review text.

Workflow:

```text
User query
 |
 v
Local SQLite product index
 |
 v
Best product candidates
 |
 v
parent_asin
 |
 v
Amazon Reviews 2023 reviews
 |
 v
Sentiment analysis
```

This allows the app to distinguish between searches such as:

```text
iPhone 12
iPhone 12 charger
RayBan black
```

---

## 12. Local Product Index

Amazon Reviews 2023 is too large to load fully in memory.

The application uses a local SQLite product index:

```text
data/amazon2023_product_index.db
```

This file is generated locally and must not be pushed to GitHub.

Build command:

```bash
python scripts/build_amazon2023_index.py --max-per-category 100000
```

Indexed categories:

```text
Cell_Phones_and_Accessories
Electronics
Clothing_Shoes_and_Jewelry
Home_and_Kitchen
Beauty_and_Personal_Care
Sports_and_Outdoors
Automotive
```

During development, around 700,000 products were indexed.

---

## 13. Smart Product Search

Main file:

```text
src/app_tools/product_index_search.py
```

The product search includes:

- Query normalization.
- French to English keyword mapping.
- Smart keyword matching.
- Accessory detection.
- Product scoring.
- Product ranking.

Examples:

```text
iPhone 12
→ Finds actual iPhone products

iPhone 12 charger
→ Finds iPhone chargers and cables

RayBan black
→ Finds Ray-Ban sunglasses
```

The search logic penalizes accessories when the user searches for the main product and boosts accessories when the user explicitly asks for them.

---

## 14. Local Reviews Index

To avoid slow retrieval from remote Amazon Reviews 2023 files, a local reviews index script was created.

Script:

```text
scripts/build_amazon2023_reviews_index.py
```

Example command:

```bash
python scripts/build_amazon2023_reviews_index.py --categories Cell_Phones_and_Accessories Clothing_Shoes_and_Jewelry --max-scan-per-category 5000000 --max-reviews-per-product 20
```

This is an optional optimization for faster Streamlit demos.

---

## 15. Human-in-the-Loop Checkpoint

Before generating the final report, the user must approve the analysis inside the Streamlit app.

If the approval checkbox is not validated, report generation is blocked.

This checkpoint is implemented in:

```text
src/orchestration/orchestrator.py
```

This satisfies the human-in-the-loop requirement.

---

## 16. Logging System

The app uses JSON logging with timestamps.

Main file:

```text
src/utils/logger.py
```

Logs are saved in:

```text
logs/agent_actions.jsonl
```

Each log contains:

```text
timestamp
agent
action
status
input
output
error
```

Logs are displayed directly inside the Streamlit interface.

---

## 17. Error Handling

Custom errors are centralized in:

```text
src/utils/errors.py
```

Main custom errors:

```text
ProductReviewError
ReviewCollectionError
SentimentAnalysisError
InsightGenerationError
HumanApprovalRequiredError
ReportGenerationError
```

This keeps the project clean, modular, and easy to defend orally.

---

## 18. Gemini API Configuration

Create a `.env` file at the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
CREWAI_LLM_MODEL=gemini/gemini-2.5-flash
```

If no Gemini API key is configured, the app still works using the fallback report generator.

---

## 19. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/fahdchaib70-blip/AI-Product-Review-Intelligence.git
cd AI-Product-Review-Intelligence
```

### Step 2 — Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

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

## 20. Run the Complete App

### Step 1 — Place the trained model folder

Place the model folder here:

```text
AI-Product-Review-Intelligence/model/sentiment_distilbert/
```

### Step 2 — Test the model

```bash
python -c "from src.sentiment_model import predict_sentiment; print(predict_sentiment('This product is amazing.'))"
```

### Step 3 — Run Streamlit

```bash
streamlit run app.py
```

### Step 4 — Use the app

1. Enter a product name.
2. Select a review source.
3. Click **Analyze Reviews**.
4. Review the sentiment results.
5. Validate the human approval checkbox.
6. Click **Generate Final Report**.
7. Read the generated product intelligence report.
8. Check logs in the logs viewer.

---

## 21. Example Demo Scenario

Recommended demo input:

```text
iPhone 12
```

Recommended review source:

```text
Amazon Reviews 2023 product-level
```

Expected app output:

- Review list.
- Positive / negative / neutral percentages.
- Sentiment chart.
- Common complaints.
- Product strengths.
- Human approval checkpoint.
- Final business report.
- JSON execution logs.

---

## 22. Requirements Files

### `requirements.txt`

Main ML dependencies:

```text
streamlit
pandas
numpy
python-dotenv
crewai
google-genai
torch
torchvision
transformers==4.53.3
tokenizers==0.21.4
datasets==3.6.0
huggingface_hub==0.36.2
scikit-learn
matplotlib
pydantic
```

---

## 23. GitHub Workflow

### Model branch

Person 1 worked on:

```text
model-part
```

Commands used:

```bash
git init
git checkout -b model-part
git add .
git commit -m "Add DistilBERT sentiment model"
git remote add origin https://github.com/fahdchaib70-blip/AI-Product-Review-Intelligence.git
git push -u origin model-part
```

### App branch

Person 2 worked on:

```text
app-part
```

Recommended commands:

```bash
git checkout model-part
git pull origin model-part
git checkout -b app-part

git add .
git commit -m "Add Streamlit multi-agent app integration"
git push -u origin app-part
```

### Merge app branch later

```bash
git checkout main
git pull origin main
git fetch origin
git merge origin/model-part
git merge origin/app-part
git push origin main
```

Alternative if `model-part` is already merged into `main`:

```bash
git checkout main
git pull origin main
git merge origin/app-part
git push origin main
```

---

## 24. Git Ignore Rules

The following files and folders should not be pushed:

```text
.venv/
.env
__pycache__/
*.pyc

model/
logs/*.jsonl
reports/*.md

data/amazon2023_product_index.db
data/amazon2023_reviews_index.db
```

Reason:

- `.venv/` is local.
- `.env` contains private API keys.
- `model/` contains large model files.
- logs and reports are generated during runtime.
- SQLite indexes are generated locally and can be large.

---

## 25. Current Status

Completed features:

```text
- Fine-tuned DistilBERT sentiment model.
- Stable predict_sentiment(text) API.
- Evaluation metrics.
- Classification report.
- Confusion matrix.
- Streamlit app.
- CrewAI agent definitions.
- Orchestrator workflow.
- Collector Agent.
- Sentiment Agent.
- Insight Agent.
- Report Agent.
- Human approval checkpoint.
- JSON logging.
- Error handling.
- Gemini report generation.
- Fallback report generation.
- Demo CSV fallback.
- Hugging Face review collector.
- Amazon Reviews 2023 product index.
- Smart product search.
- Product-level review collection.
```

---

## 26. Current Limitations

### Model storage

The trained model is not stored on GitHub because the model file is larger than GitHub's file size limit.

Solution:

```text
Share model/sentiment_distilbert/ separately.
```

### Amazon Reviews 2023 retrieval speed

The product search is fast because it uses a local SQLite product index.

However, review retrieval can be slow if reviews are streamed from remote Amazon Reviews 2023 files.

Recommended improvement:

```text
Use scripts/build_amazon2023_reviews_index.py to build a local reviews index.
```

---

## 27. Oral Defense Summary

This project is a complete AI product review intelligence system.

The ML part fine-tunes a DistilBERT model on product reviews to classify customer sentiment. The model achieves around 93% validation accuracy and exposes a clean `predict_sentiment(text)` function.

The application part uses a multi-agent architecture. The Collector Agent collects reviews, the Sentiment Agent applies the fine-tuned model, the Insight Agent extracts business indicators, and the Report Agent generates a final report using Gemini API. The Orchestrator coordinates the full workflow and blocks report generation until the human-in-the-loop checkpoint is validated.

The system includes JSON logging, centralized error handling, a Streamlit interface, CrewAI agent definitions, Gemini report generation, multiple review sources, and a modular architecture that is easy to explain and extend.

---

## 28. Short Presentation Explanation

The AI Product Review Intelligence System analyzes customer reviews using a fine-tuned DistilBERT model and a multi-agent architecture.

The workflow starts in Streamlit, where the user enters a product name. The Collector Agent retrieves reviews from manual input, demo CSV, Hugging Face datasets, or Amazon Reviews 2023. The Sentiment Agent uses the fine-tuned DistilBERT model to classify each review. The Insight Agent calculates positive, negative, and neutral percentages, then extracts common complaints and product strengths. Before the final report is created, the user must approve the analysis through a human-in-the-loop checkpoint. After approval, the Report Agent generates a structured product intelligence report using Gemini API or a fallback generator.

This architecture satisfies all project requirements: fine-tuned deep learning model, multi-agent collaboration, orchestrator, Streamlit interface, human validation, logging, error handling, and report generation.

---

## 29. Useful Commands Summary

```bash
# Clone repository
git clone https://github.com/fahdchaib70-blip/AI-Product-Review-Intelligence.git
cd AI-Product-Review-Intelligence

# Create environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test model
python -c "from src.sentiment_model import predict_sentiment; print(predict_sentiment('This product is amazing.'))"

# Run app
streamlit run app.py

# Build Amazon product index
python scripts/build_amazon2023_index.py --max-per-category 100000

# Optional: build reviews index
python scripts/build_amazon2023_reviews_index.py --categories Cell_Phones_and_Accessories Clothing_Shoes_and_Jewelry --max-scan-per-category 5000000 --max-reviews-per-product 20
```

---

