# E-Consultation Feedback Sentiment Analysis Platform 🔭

An enterprise-grade, end-to-end Machine Learning web application designed to analyze patient and provider feedback collected through e-consultation modules. Built with **Flask**, **Scikit-Learn**, **NLTK**, **MySQL**, and **Chart.js**, the platform delivers real-time sentiment inference, automated bulk dataset processing, interactive analytics dashboards, and historical record tracking.

---

## 📑 Table of Contents
- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Machine Learning Pipeline & Training](#machine-learning-pipeline--training)
- [MySQL Database Integration](#mysql-database-integration)
- [Frontend & Visualization System](#frontend--visualization-system)
- [Flask Application Routes](#flask-application-routes)
- [Project Directory Structure](#project-directory-structure)
- [Setup & Run Guide](#setup--run-guide)
- [Verification & Testing](#verification--testing)

---

## 🏗️ Architecture Overview

The system processes feedback through a multi-tiered pipeline:

```
[Raw User Feedback / CSV Upload]
               │
               ▼
   [NLTK Preprocessing Pipeline]
   ├── Lowercasing & regex cleaning
   ├── Contraction normalization
   ├── Negation-word preservation
   ├── English stopword filtering
   └── Dual-pass lemmatization (Verb + Noun)
               │
               ▼
     [TF-IDF Feature Extractor]
     ├── Sublinear Term Frequency
     └── Unigram & Bigram Features (max 5,000)
               │
               ▼
   [Calibrated Linear SVM Classifier]  ── (Parallel VADER Lexicon Baseline)
               │
               ▼
   [Sentiment Decision & Confidence]
   ├── Positive (> 0.55 probability)
   ├── Negative (< 0.45 probability)
   └── Neutral (45% - 55% uncertainty window)
               │
       ┌───────┴───────┐
       ▼               ▼
[MySQL Database] [Interactive UI & Chart.js Dashboard]
```

---

## 💻 Technology Stack

### 1. Backend & Web Framework
- **Python (3.8 - 3.14)**: Core backend programming language.
- **Flask (>= 3.0.0)**: Lightweight, modular WSGI web application framework managing routes, template rendering, and RESTful endpoints.
- **Python-Dotenv**: Environment variable configuration management via `.env`.

### 2. Natural Language Processing & Machine Learning
- **NLTK (>= 3.8.1)**:
  - Tokenization via `word_tokenize` / `punkt` tokenizer.
  - Stopword filtering with explicit negation preservation (`not`, `never`, `cannot`, etc.).
  - Lemmatization via `WordNetLemmatizer` across verb and noun parts-of-speech.
- **Scikit-Learn (>= 1.4.0)**:
  - `TfidfVectorizer`: Sublinear TF-IDF feature extraction with n-gram range `(1, 2)`.
  - `LinearSVC`: High-performance support vector classifier for text classification.
  - `CalibratedClassifierCV`: Platt scaling probability calibration for confidence scoring.
  - `LogisticRegression` & `MultinomialNB`: Baseline candidate benchmark models.
- **vaderSentiment (>= 3.3.2)**: Valence Aware Dictionary for sEntiment Reasoning, providing a rule-based baseline comparison for every prediction.
- **Joblib (>= 1.3.0)**: High-performance serialization and loading of trained model and vectorizer artifacts.
- **Pandas (>= 2.0.0) & NumPy (>= 1.24.0)**: High-performance dataset parsing, batch transformation, and metric computation.

### 3. Database Layer
- **MySQL Server (5.7 / 8.0 / MariaDB)**: Relational database storing all processed comments, confidence metrics, timestamps, and metadata.
- **PyMySQL (>= 1.1.0)**: Pure Python MySQL client with dictionary cursor support and connection diagnostics.

### 4. Frontend & User Interface
- **HTML5 & Vanilla CSS3**: Semantic markup and modern design system (`static/css/style.css`) featuring:
  - Glassmorphic panels and subtle gradients
  - Responsive KPI cards with percentage distributions
  - Live prediction badge indicators with dynamic confidence progress meters
- **Vanilla JavaScript (ES6+)**: Asynchronous fetch handlers (`static/js/dashboard.js`), dynamic DOM injection, tab switching, and chart refresh triggers.
- **Chart.js (4.4.x)**: Responsive, canvas-rendered interactive data visualizations.

---

## 🔬 Machine Learning Pipeline & Training

### Dataset
The model is trained on `econsult_comments_dataset.csv`, consisting of 400 labeled consultation comments balanced across categories (General, Medication, Technical, Usability, Accessibility, Chronic Care, etc.).

### Training Workflow (`train_model.py`)
1. **Data Ingestion**: Loads comments and sentiment labels, cleaning null entries.
2. **Text Preprocessing**: Runs each sample through `preprocess.py` (negation-preserving stopword removal and lemmatization).
3. **Stratified Split**: 80% training set (320 samples) and 20% test set (80 samples) preserving class balance.
4. **Vectorization**: Fits a TF-IDF vectorizer on training text:
   - `ngram_range=(1, 2)` (captures phrases like "not recommend" or "very helpful")
   - `sublinear_tf=True` (logarithmic term frequency scaling)
   - `max_features=5000`
5. **Model Evaluation & Benchmarking**: Evaluates three candidate classifiers:

| Classifier Candidate | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **93.75%** | **93.78%** | **93.75%** | **93.75%** |
| Logistic Regression | 93.75% | 93.77% | 93.75% | 93.75% |
| Multinomial Naive Bayes | 92.50% | 92.50% | 92.50% | 92.50% |

6. **Artifact Serialization**: The winning model, TF-IDF vectorizer, and benchmarking metrics are saved to `models/`:
   - `models/sentiment_model.joblib`
   - `models/tfidf_vectorizer.joblib`
   - `models/model_metrics.json`

---

## 🗄️ MySQL Database Integration

The application integrates with MySQL via `db.py`, featuring:
- **Port Availability Pre-check**: Fast non-blocking socket test (`is_port_open`) to diagnose connection issues before database timeout.
- **Automatic Initialization**: Auto-creates the `sentiment_analysis` database and `comments` table upon first connection.
- **Database Schema**:

```sql
CREATE TABLE IF NOT EXISTS comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    comment_text TEXT NOT NULL,
    domain VARCHAR(100) DEFAULT 'general',
    sentiment VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    vader_sentiment VARCHAR(20) DEFAULT NULL,
    vader_compound FLOAT DEFAULT NULL,
    source VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

- **Connection Status Caching**: 5-second health check cache prevents latency overhead on repeated requests.

---

## 📊 Frontend & Visualization System

The interactive frontend offers three dedicated views powered by Jinja2 templates:

### 1. Dashboard (`templates/index.html`)
- **KPI Cards**: Real-time counters for Total Comments, Positive Feedback %, Negative Feedback %, and Neutral/Borderline %.
- **Live Comment Tester**: Instant inference form with AJAX submission, displaying:
  - Primary ML classification badge (`POSITIVE`, `NEGATIVE`, `NEUTRAL`)
  - Confidence percentage and dynamic progress bar
  - NLTK preprocessed tokens
  - VADER lexicon comparative baseline (compound score, positive, negative, and neutral ratios)
- **Visual Analytics Tabs (Chart.js)**:
  - *Distribution Chart*: Doughnut chart illustrating sentiment proportions.
  - *Domain Breakdown*: Stacked horizontal/vertical bar chart showing sentiment distributions across medical domains.
  - *Model Benchmarks*: Comparative bar chart plotting Accuracy, Precision, Recall, and F1 across trained algorithms.
- **Recent Comments Feed**: Live-updated table displaying the latest comments analyzed.

### 2. Bulk CSV Analysis (`templates/bulk.html`)
- Drag-and-drop CSV upload zone supporting batch feedback processing.
- Instant evaluation using `test_comments.csv` via the "Test with Sample Dataset" button.
- Summary KPI card and paginated result table with sentiment badges and confidence ratings.

### 3. History & Records (`templates/history.html`)
- Comprehensive table displaying all stored records from MySQL.
- Keyword search filter (searches within feedback text).
- Sentiment dropdown filter (`All`, `Positive Only`, `Negative Only`, `Neutral Only`).
- Individual record deletion button with confirmation dialog.
- CSV Export action button.

---

## 🚦 Flask Application Routes

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Renders the primary analytics dashboard with live KPI counters, charts, and recent records. |
| `POST` | `/predict` | Predicts sentiment for a single comment. Accepts JSON (`{"comment": "...", "domain": "..."}`) or form data. Returns prediction, confidence, preprocessed tokens, and VADER scores. |
| `GET` | `/bulk` | Renders the batch CSV upload and analysis interface. |
| `POST` | `/predict-bulk` | Processes uploaded CSV file containing a `comment` column, performs batch inference, and persists records to MySQL. |
| `GET` | `/load-sample` | Convenience endpoint that loads and analyzes `test_comments.csv`, populating batch results and MySQL. |
| `GET` | `/history` | Displays stored database records with search and sentiment filtering. |
| `POST` | `/history/delete/<id>` | Deletes a comment record by its primary key ID. |
| `GET` | `/api/stats` | JSON endpoint returning aggregated KPI statistics and domain distribution for dynamic Chart.js rendering. |
| `GET` | `/export` | Streams all stored database comments as a downloadable CSV (`econsult_sentiment_records.csv`). |

---

## 📁 Project Directory Structure

```
sentiment-analysis-of-comments-received-through-e-consultation-module/
├── app.py                         # Main Flask application and routing
├── db.py                          # MySQL connection manager, schema init, and CRUD
├── preprocess.py                  # NLTK text cleaner, lemmatizer, and VADER baseline
├── train_model.py                 # Model training, evaluation, and serialization script
├── requirements.txt               # Python package dependencies
├── .env                           # Database credentials and application secrets
├── .env.example                   # Template for environment configuration
├── econsult_comments_dataset.csv  # 400-comment training dataset
├── test_comments.csv              # Sample batch dataset for testing bulk upload
├── models/
│   ├── sentiment_model.joblib     # Calibrated Linear SVM serialized model
│   ├── tfidf_vectorizer.joblib    # Fitted TF-IDF vectorizer
│   └── model_metrics.json         # Benchmark metrics and evaluation results
├── static/
│   ├── css/
│   │   └── style.css              # Custom CSS design system and responsive layout
│   └── js/
│       └── dashboard.js           # Chart.js initialization and AJAX form handlers
└── templates/
    ├── base.html                  # Core Jinja2 base layout with navigation & alerts
    ├── index.html                 # Analytics dashboard & single prediction interface
    ├── bulk.html                  # Batch CSV upload and processing interface
    ├── history.html               # Records table with search, filter, and export
    └── db_error.html              # Diagnostic error page for MySQL disconnection
```

---

## 🚀 Setup & Run Guide

### 1. Prerequisites
- **Python 3.8 to 3.14**
- **MySQL Server** (Running locally via XAMPP, WAMP, Docker, or native Windows service)

### 2. Clone Repository & Setup Virtual Environment
```bash
# Navigate to the project folder
cd sentiment-analysis-of-comments-received-through-e-consultation-module

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (Linux / macOS)
# source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Database (`.env`)
Create or verify `.env` in the root directory:
```ini
# MySQL Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=sentiment_analysis

# Flask Configuration
SECRET_KEY=econsult_sentiment_secret_key_2026
PORT=5000
```
> **Note**: If the `sentiment_analysis` database does not exist, the application will automatically create it upon launch.

### 5. Train the Machine Learning Model (Optional)
Pre-trained model artifacts are already included in `models/`. To retrain the models from scratch:
```bash
python train_model.py
```

### 6. Start the Flask Application
```bash
python app.py
```
The server will start at **http://127.0.0.1:5000**. Open your browser to access the dashboard.

---

## ✅ Verification & Testing

Verify that all components are functioning correctly:

1. **Dashboard & KPIs**: Open `http://127.0.0.1:5000/` and ensure KPI cards show the stored totals and Chart.js charts render without errors.
2. **Single Comment Inference**: Enter feedback into the tester (e.g. *"Doctor was punctual and medication was explained clearly"*) and click **Analyze Sentiment**. Verify that the result card displays the sentiment, confidence bar, preprocessed tokens, and VADER baseline.
3. **Bulk CSV Processing**: Navigate to **Bulk CSV Analysis** (`/bulk`), click **Test with Sample Dataset**, and verify that the 10 comments from `test_comments.csv` are processed and displayed.
4. **History & Search**: Navigate to **History & Records** (`/history`), test filtering by sentiment (*Positive Only*), search for keywords (*e.g., "doctor"*), and confirm records filter accordingly.
5. **CSV Export**: Click **Export to CSV** on the History page or visit `/export` to download `econsult_sentiment_records.csv`.
6. **API Verification**: Visit `http://127.0.0.1:5000/api/stats` to confirm JSON output from MySQL.

---

## 📄 License
This project is licensed under the MIT License.
