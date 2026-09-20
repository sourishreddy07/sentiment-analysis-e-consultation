# Sentiment Analysis of Comments Received Through E-Consultation Module

A Flask-based machine learning web application for analyzing patient and provider feedback received through electronic consultation modules. The platform utilizes Natural Language Processing (NLP) text preprocessing, TF-IDF feature extraction, and supervised machine learning classification, backed by MySQL relational storage and an interactive web dashboard for real-time analytics.

---

## Project Overview

Electronic consultation (e-consultation) systems allow healthcare users to submit inquiries, service ratings, and unstructured feedback regarding their consultation experience. As consultation volumes grow, manually reviewing and categorizing feedback becomes inefficient and labor-intensive.

- **Problem Addressed:** High volumes of unstructured textual feedback make manual evaluation slow and inconsistent across healthcare domains.
- **Objective:** Provide an automated, interpretable pipeline that processes feedback, predicts sentiment, quantifies classification confidence, and organizes records into domain-specific analytics.
- **Feedback Summarization:** Automated sentiment classification enables administrative and clinical teams to rapidly identify operational issues, track satisfaction trends, and prioritize negative feedback for service improvement.
- **Workflow Flexibility:** The system supports both real-time individual comment inference for interactive testing and batch CSV processing for historical feedback datasets.

---

## Key Features

- **Single-Comment Sentiment Prediction:** Real-time sentiment classification of individual comments with instant probability feedback.
- **NLP Preprocessing Pipeline:** Text cleaning, lowercasing, URL/HTML tag removal, contraction expansion, negation preservation, stopword removal, and dual-pass lemmatization using NLTK.
- **TF-IDF Feature Extraction:** Word representation using unigram and bigram features with sublinear term frequency scaling.
- **Supervised ML Classification:** Trained on domain-specific e-consultation feedback with empirical evaluation.
- **Model Comparison:** Evaluates Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine (Linear SVM).
- **Confidence Scoring:** Outputs classification probability/confidence score for model transparency.
- **Neutral / Uncertainty Handling:** Predictions falling within an uncertainty margin (45%–55% positive class probability) are categorized as neutral at the application level.
- **VADER Baseline Comparison:** Computes rule-based VADER lexicon compound scores in parallel for comparative benchmarking.
- **Bulk CSV Sentiment Analysis:** Ingests CSV feedback datasets, runs batch inference, computes aggregate statistics, and displays tabular results.
- **MySQL Persistence:** Relational database storage for all analyzed comments, sentiment tags, confidence scores, domains, and timestamps.
- **Interactive Dashboard:** Visual KPI overview displaying total comments, positive, negative, and neutral percentages.
- **Sentiment Distribution Visualization:** Dynamic Chart.js charts showing overall feedback distribution.
- **Domain-Based Analytics:** Categorization across consultation domains (medication, technical, general, usability, accessibility, etc.).
- **Prediction History:** Paginated table of stored feedback records.
- **Search and Filtering:** Real-time text search and multi-criteria filtering by domain, sentiment, and session.
- **CSV Export:** Download analyzed comments and prediction metadata directly from the database into CSV format.
- **Demo Session Isolation:** Automatic UUID-based session tracking (`demo_session_id`) allows demonstration runs to showcase live counters without losing persistent database history.

---

## System Architecture

```
User / CSV Feedback
        ↓
Flask Web Application
        ↓
Text Preprocessing (NLTK)
        ↓
TF-IDF Feature Extraction (Unigram + Bigram)
        ↓
Machine Learning Model (Linear SVM / Calibrated)
        ↓
Sentiment + Confidence Score
        ↓
MySQL Database (Persistent Storage)
        ↓
Dashboard / History / Analytics
```

> **Note:** Alongside the supervised ML prediction, VADER lexicon compound polarity is computed in parallel to provide a rule-based baseline comparison.

---

## Machine Learning Pipeline

1. **Data Loading:** Reads `econsult_comments_dataset.csv` into a Pandas DataFrame.
2. **Missing-Value Handling:** Strips whitespace and removes null records in comment text or sentiment labels.
3. **NLTK Preprocessing:** Normalizes text by removing URLs, stripping HTML tags, expanding common English contractions, preserving negation keywords (e.g., *not*, *never*, *cannot*), eliminating standard stopwords, and applying dual-pass lemmatization (verb followed by noun).
4. **TF-IDF Vectorization:** Converts preprocessed tokens into numerical features using `TfidfVectorizer` configured for unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency, and up to 5,000 maximum features.
5. **Stratified Train/Test Split:** Partitions data into an 80% training set and a 20% testing set using stratified sampling to preserve class proportions.
6. **Candidate Model Training:** Fits three supervised classifiers:
   - Logistic Regression (`max_iter=1000`)
   - Multinomial Naive Bayes
   - Linear SVM (`LinearSVC` wrapped in `CalibratedClassifierCV` for well-calibrated probability estimates)
7. **Evaluation:** Measures accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrix on the held-out test split.
8. **Best Model Selection:** Automatically selects the best-performing model based on the measured test F1-score.
9. **Artifact Serialization:** Persists the trained model (`sentiment_model.joblib`), the fitted vectorizer (`tfidf_vectorizer.joblib`), and empirical evaluation metrics (`model_metrics.json`) into the `models/` directory.
10. **Application Inference:** The Flask backend loads the serialized model and vectorizer at startup to perform live probability scoring and threshold-based classification.

---

## Dataset

- **Dataset File:** `econsult_comments_dataset.csv`
- **Total Records:** 400
- **Class Distribution:**
  - Positive: 206 (51.5%)
  - Negative: 194 (48.5%)
- **Train/Test Split:** 80/20 stratified split
- **Training Samples:** 320
- **Test Samples:** 80

The training dataset is a binary-labeled dataset (positive and negative feedback). The training pipeline does not treat neutral as a third training class. Instead, the application layer implements an uncertainty window: when a comment's predicted positive probability falls between 0.45 and 0.55, the application reports a "neutral" classification to signify borderline sentiment.

---

## Model Evaluation

The candidate models were evaluated on the 80-sample held-out test split using `train_model.py`. The recorded metrics from `models/model_metrics.json` are:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Linear SVM** | **93.75%** | **93.78%** | **93.75%** | **93.75%** |
| **Logistic Regression** | 93.75% | 93.77% | 93.75% | 93.75% |
| **Multinomial Naive Bayes** | 92.50% | 92.50% | 92.50% | 92.50% |

### Key Findings
- **Linear SVM** was selected by the pipeline as the primary deployment model because it achieved the highest measured test F1-score (93.75%) and highest weighted precision (93.78%).
- **Logistic Regression** produced an identical F1-score (93.75%) and recall (93.75%) with slightly lower precision (93.77%).
- **Multinomial Naive Bayes** achieved competitive results with an F1-score of 92.50%.
- *Note:* These metrics reflect empirical performance on the project's 80/20 test split and are not presented as universal performance across arbitrary external domains.

---

## Technology Stack

### Backend
- **Python** (Core application runtime)
- **Flask** (WSGI web application framework and routing)

### Machine Learning & NLP
- **Scikit-learn** (Model training, evaluation metrics, probability calibration)
- **NLTK** (Tokenization, stopword filtering, WordNet lemmatization)
- **TF-IDF** (Text feature vectorization)
- **Logistic Regression** (Candidate classifier)
- **Multinomial Naive Bayes** (Candidate classifier)
- **Linear SVM** (Selected classifier via `LinearSVC`)
- **VADER** (`vaderSentiment` lexicon baseline comparison)

### Data Handling & Persistence
- **Pandas** (Dataset ingestion and tabular transformations)
- **NumPy** (Numerical operations and array handling)
- **Joblib** (Model and vectorizer serialization)

### Database
- **MySQL** (Relational storage for comments and predictions)
- **PyMySQL** (Python MySQL database client driver)

### Frontend
- **HTML5 & CSS3** (Semantic structure and responsive styling)
- **JavaScript (ES6+)** (Asynchronous DOM updates and API calls)
- **Chart.js** (Interactive data visualization charts)
- **Jinja2** (Server-side template rendering)

### Configuration
- **python-dotenv** (Environment variable management)

---

## Project Structure

```
sentiment-analysis-e-consultation/
├── app.py
├── db.py
├── preprocess.py
├── train_model.py
├── test_verify.py
├── requirements.txt
├── .env.example
├── .gitignore
├── econsult_comments_dataset.csv
├── test_comments.csv
├── models/
│   ├── sentiment_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metrics.json
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── dashboard.js
└── templates/
    ├── base.html
    ├── index.html
    ├── bulk.html
    ├── history.html
    └── db_error.html
```

---

## Installation

### 1. Clone the Repository
Open Windows PowerShell and clone the project:

```powershell
git clone https://github.com/sourishreddy07/sentiment-analysis-e-consultation.git
cd sentiment-analysis-e-consultation
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Execution Policy Note:** If PowerShell displays an `Execution_Policies` restriction error, you can bypass it for the current terminal session:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```
> Alternatively, activate using the standard Command Prompt (`cmd.exe`):
> ```cmd
> .venv\Scripts\activate.bat
> ```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Environment Configuration

The repository excludes local configuration files (`.env`) from version control for security.

1. Copy the example configuration template:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and set your local database parameters:
   ```ini
   # MySQL Database Configuration
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_mysql_password
   DB_NAME=sentiment_analysis

   # Flask Secret Key
   SECRET_KEY=your_random_secret_key
   ```

---

## MySQL Setup

1. Ensure the MySQL server is running (e.g., via XAMPP Control Panel, MySQL Workbench, WampServer, or the native Windows MySQL service).
2. The application includes automated database initialization in `db.py`. When `app.py` runs (or when `db.init_db()` is executed), it automatically creates:
   - Database: `sentiment_analysis` (if it does not exist)
   - Table: `comments` with schema for comment text, domain, sentiment, confidence, VADER scores, session identifier, and timestamps.
3. *MySQL is required for data persistence and history tracking.* If MySQL is unreachable, the application displays a descriptive connection diagnosis page (`templates/db_error.html`).

---

## Model Training

To retrain the machine learning models and regenerate evaluation artifacts:

```powershell
python train_model.py
```

This script:
1. Loads `econsult_comments_dataset.csv`.
2. Applies NLTK cleaning, stopword filtering, and lemmatization.
3. Fits the unigram/bigram TF-IDF vectorizer.
4. Trains Logistic Regression, Multinomial Naive Bayes, and Linear SVM classifiers.
5. Evaluates models on the 80-sample test split and prints confusion matrices.
6. Selects the model with the highest test F1-score.
7. Saves `sentiment_model.joblib`, `tfidf_vectorizer.joblib`, and `model_metrics.json` into the `models/` directory.

---

## Run the Application

Start the Flask development server:

```powershell
python app.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Testing & Verification

### Automated Verification Script
Run the built-in verification suite:

```powershell
python test_verify.py
```

`test_verify.py` performs the following validation checks:
- Confirms the model artifact is loaded and is an active classifier instance.
- Confirms the TF-IDF vectorizer artifact is loaded.
- Evaluates a sample input comment through the Flask inference function `predict_sentiment_single()`.
- Computes direct probability outputs from the vectorizer and model pipeline.
- Asserts that prediction outcomes and confidence scores match with precision `< 1e-4`.

### Manual System Verification
- **Dashboard:** Open `http://127.0.0.1:5000` to verify KPI cards and Chart.js distribution charts.
- **Single Comment Tester:** Submit a sample comment on the dashboard and verify the returned sentiment badge, confidence bar, preprocessed tokens, and VADER baseline score.
- **Bulk CSV Analysis:** Navigate to `/bulk`, upload `test_comments.csv` (or use the sample load button), and confirm that batch analytics render.
- **History View:** Navigate to `/history` to inspect persisted comments, execute text searches, and test domain filters.
- **CSV Export:** Click "Export CSV" on the history page and verify that `econsult_sentiment_records.csv` downloads with complete records.
- **MySQL Status:** Check the top navigation bar to verify the MySQL connection status indicator.
- **API Endpoint:** Access `http://127.0.0.1:5000/api/stats` to verify JSON statistics output.

---

## API Endpoints

The Flask application exposes the following routes:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the interactive dashboard with KPI cards, single-comment tester, and Chart.js visualizations. |
| `POST` | `/predict` | Analyzes a single comment from JSON or form data; returns predicted sentiment, confidence score, preprocessed tokens, and VADER scores; stores record in MySQL. |
| `GET` | `/bulk` | Renders the batch CSV upload and analysis interface. |
| `POST` | `/predict-bulk` | Processes an uploaded CSV file, executes batch sentiment prediction, stores records in MySQL, and returns aggregated batch metrics. |
| `POST` | `/load-sample` | Loads bundled sample records from `test_comments.csv` into the active session for demonstration. |
| `GET` | `/history` | Displays the historical feedback log with pagination, text search, domain filtering, and sentiment filtering. |
| `POST` | `/history/delete/<int:record_id>` | Deletes an individual analyzed comment record from MySQL by its primary key ID. |
| `GET` | `/api/stats` | Returns a JSON payload of current statistics (total, positive, negative, neutral counts, percentages, and domain breakdown). |
| `GET` | `/export` | Queries all stored feedback records from MySQL and streams them as a downloadable CSV file. |

---

## Security Notes

- **Environment Separation:** The `.env` file is excluded in `.gitignore` to prevent leaking database credentials and secret keys.
- **Template Placeholders:** `.env.example` contains only placeholder configurations and no actual secrets.
- **Virtual Environments:** Virtual environment folders (`.venv/` and `venv/`) are ignored and should never be committed.
- **Secret Key:** The `SECRET_KEY` variable must be changed to a cryptographically secure random value before production deployment.
- **Scope Disclaimer:** This application is an academic/research demonstration system and is not certified for clinical diagnostics or medical decision-making.

---

## Limitations

- **Dataset Size:** The model is trained on a curated dataset of 400 domain-specific comments; performance on broader, out-of-domain text may vary.
- **Binary Training Annotations:** Ground-truth training labels are binary (positive and negative). Neutral classifications reflect application-level uncertainty boundaries (45%–55% probability) rather than a separately annotated training class.
- **Evaluation Context:** Reported metrics are derived from the project's 80/20 train/test split.
- **Database Dependency:** MySQL is required for persistent record storage, historical auditing, and dashboard metrics.

---

## Future Enhancements

- **Dataset Expansion:** Collecting larger, multi-facility consultation feedback datasets to capture broader medical terminology and patient concerns.
- **Multilingual Support:** Implementing cross-lingual preprocessing to analyze consultation feedback submitted in multiple regional languages.
- **Transformer Architectures:** Evaluating domain-specific transformer models (e.g., ClinicalBERT, BioBERT) for enhanced contextual representation.
- **Model Monitoring:** Adding data drift detection and automated retraining pipelines as new feedback accumulates.
- **Role-Based Access Control:** Adding user authentication and audit logging for administrative governance.
- **Containerization & Deployment:** Dockerizing the Flask and MySQL services for automated deployment.
- **Aspect-Based Sentiment Analysis:** Identifying sentiment toward specific aspects of care (e.g., wait times, prescription clarity, physician empathy).

---

## Academic Project Note

This project was developed as an academic engineering demonstration showcasing an end-to-end applied machine learning workflow:

$$\text{Data Preprocessing (NLTK)} \longrightarrow \text{TF-IDF Feature Extraction} \longrightarrow \text{Supervised Classification (Linear SVM)} \longrightarrow \text{Flask Service} \longrightarrow \text{MySQL} \longrightarrow \text{Analytics Dashboard}$$

It is intended for educational and research evaluation and does not claim clinical validation.

---

## License

License information will be added in a future revision.
