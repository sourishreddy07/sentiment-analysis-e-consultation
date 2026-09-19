"""
Flask Web Application for E-Consultation Feedback Sentiment Analysis.

Features:
- NLTK preprocessing pipeline
- TF-IDF feature extraction
- Scikit-Learn supervised classifier
- VADER baseline comparison
- MySQL persistence
- Chart.js dashboard support
- Batch CSV sentiment analysis
- Prediction history
- CSV export
"""

import io
import json
import os

import joblib
import pandas as pd

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from dotenv import load_dotenv

from preprocess import (
    preprocess_comment,
    get_vader_sentiment,
)

import db


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "econsult_sentiment_secret_key_2026"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "sentiment_model.joblib"
)

VECTORIZER_PATH = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer.joblib"
)

METRICS_PATH = os.path.join(
    MODELS_DIR,
    "model_metrics.json"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

model = None
vectorizer = None

try:

    # --------------------------------------------------------
    # Check whether trained model exists
    # --------------------------------------------------------

    if (
        not os.path.exists(MODEL_PATH)
        or not os.path.exists(VECTORIZER_PATH)
    ):

        print(
            "[!] Trained model/vectorizer not found."
        )

        try:

            from train_model import train_and_evaluate

            print(
                "[*] Starting model training..."
            )

            train_and_evaluate()

        except Exception as training_error:

            print(
                "[!] Automatic model training failed: "
                f"{training_error}"
            )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        model = joblib.load(
            MODEL_PATH
        )

    # --------------------------------------------------------
    # Load TF-IDF vectorizer
    # --------------------------------------------------------

    if os.path.exists(VECTORIZER_PATH):

        vectorizer = joblib.load(
            VECTORIZER_PATH
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if (
        model is not None
        and vectorizer is not None
    ):

        print(
            "[+] ML model and TF-IDF vectorizer loaded."
        )

    else:

        print(
            "[!] ML model/vectorizer unavailable."
        )

except Exception as error:

    model = None
    vectorizer = None

    print(
        "[!] Error loading ML model/vectorizer: "
        f"{error}"
    )


# ============================================================
# LOAD MODEL METRICS
# ============================================================

metrics_data = {}

if os.path.exists(METRICS_PATH):

    try:

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metrics_data = json.load(file)

        print(
            "[+] Model metrics loaded."
        )

    except Exception as error:

        print(
            "[!] Error loading model metrics: "
            f"{error}"
        )


# ============================================================
# DATABASE STATUS
# ============================================================

def get_db_status():
    """
    Check MySQL connection status.
    """

    try:

        connected, message = (
            db.check_connection()
        )

        return {
            "connected": connected,
            "message": message
        }

    except Exception as error:

        return {
            "connected": False,
            "message": str(error)
        }


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

db_status = get_db_status()

if db_status["connected"]:

    try:

        db.init_db()

        print(
            "[+] MySQL database and comments table ready."
        )

    except Exception as error:

        print(
            "[!] MySQL initialization error: "
            f"{error}"
        )

else:

    print(
        "[!] MySQL not connected: "
        f"{db_status['message']}"
    )


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment_single(comment_text: str):
    """
    Perform sentiment prediction.

    Pipeline:

    Comment
        ↓
    NLTK preprocessing
        ↓
    TF-IDF
        ↓
    Scikit-Learn Model
        ↓
    Sentiment + Confidence

    VADER is also calculated as a baseline.
    """

    # --------------------------------------------------------
    # PREPROCESS COMMENT
    # --------------------------------------------------------

    cleaned = preprocess_comment(
        comment_text
    )

    sentiment = None
    confidence = 0.0

    # --------------------------------------------------------
    # PRIMARY ML MODEL
    # --------------------------------------------------------

    if (
        model is not None
        and vectorizer is not None
    ):

        try:

            X_vec = vectorizer.transform(
                [cleaned]
            )

            # ------------------------------------------------
            # Models with predict_proba()
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(X_vec)[0]
                )

                classes = list(
                    model.classes_
                )

                # --------------------------------------------
                # Find positive class
                # --------------------------------------------

                if "positive" in classes:

                    positive_index = (
                        classes.index("positive")
                    )

                else:

                    positive_index = None

                # --------------------------------------------
                # Find negative class
                # --------------------------------------------

                if "negative" in classes:

                    negative_index = (
                        classes.index("negative")
                    )

                else:

                    negative_index = None

                # --------------------------------------------
                # Positive probability
                # --------------------------------------------

                positive_probability = (

                    float(
                        probabilities[
                            positive_index
                        ]
                    )

                    if positive_index is not None

                    else 0.0
                )

                # --------------------------------------------
                # Negative probability
                # --------------------------------------------

                negative_probability = (

                    float(
                        probabilities[
                            negative_index
                        ]
                    )

                    if negative_index is not None

                    else 0.0
                )

                # --------------------------------------------
                # Sentiment decision
                # --------------------------------------------

                if (
                    0.45
                    <= positive_probability
                    <= 0.55
                ):

                    sentiment = "neutral"

                    confidence = round(
                        max(
                            positive_probability,
                            negative_probability
                        ),
                        4
                    )

                elif positive_probability > 0.55:

                    sentiment = "positive"

                    confidence = round(
                        positive_probability,
                        4
                    )

                else:

                    sentiment = "negative"

                    confidence = round(
                        negative_probability,
                        4
                    )

            # ------------------------------------------------
            # Models without predict_proba()
            # ------------------------------------------------

            else:

                prediction = model.predict(
                    X_vec
                )[0]

                sentiment = str(
                    prediction
                ).lower()

                # A classifier such as LinearSVC does not
                # directly provide probabilities.
                #
                # We keep the existing behavior of the
                # application and use a display confidence.

                confidence = 0.85

        except Exception as error:

            print(
                "[!] ML prediction error: "
                f"{error}"
            )

            sentiment = None
            confidence = 0.0

    # --------------------------------------------------------
    # VADER FALLBACK
    # --------------------------------------------------------

    if sentiment is None:

        vader_result = get_vader_sentiment(
            comment_text
        )

        sentiment = vader_result[
            "sentiment"
        ]

        confidence = abs(
            float(
                vader_result["compound"]
            )
        )

    # --------------------------------------------------------
    # VADER BASELINE
    # --------------------------------------------------------

    vader = get_vader_sentiment(
        comment_text
    )

    return {
        "sentiment": sentiment,

        "confidence": round(
            float(confidence),
            4
        ),

        "preprocessed_comment": cleaned,

        "vader": vader
    }


# ============================================================
# DASHBOARD
# ============================================================


def empty_stats():
    """Return the dashboard payload used when the app is first opened."""
    return {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "positive_pct": 0,
        "negative_pct": 0,
        "neutral_pct": 0,
        "domain_distribution": [],
        "recent_comments": []
    }


@app.route("/")
def index():

    status = get_db_status()
    stats = empty_stats()

    return render_template(
        "index.html",
        active_page="dashboard",
        db_status=status,
        stats=stats,
        metrics=metrics_data
    )


# ============================================================
# SINGLE COMMENT PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # JSON request
    # --------------------------------------------------------

    if request.is_json:

        data = request.get_json() or {}

        comment = str(
            data.get(
                "comment",
                ""
            )
        ).strip()

        domain = str(
            data.get(
                "domain",
                "general"
            )
        ).strip().lower()

    # --------------------------------------------------------
    # Normal form request
    # --------------------------------------------------------

    else:

        comment = request.form.get(
            "comment",
            ""
        ).strip()

        domain = request.form.get(
            "domain",
            "general"
        ).strip().lower()

    # --------------------------------------------------------
    # Validate comment
    # --------------------------------------------------------

    if not comment:

        return jsonify(
            {
                "success": False,
                "error": "Comment cannot be empty."
            }
        ), 400

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_sentiment_single(
        comment
    )

    # --------------------------------------------------------
    # SAVE TO MYSQL
    # --------------------------------------------------------

    db_saved = False
    db_warning = None
    inserted_id = None

    status = get_db_status()

    if status["connected"]:

        try:

            inserted_id = db.insert_comment(

                comment_text=comment,

                domain=domain,

                sentiment=prediction[
                    "sentiment"
                ],

                confidence=prediction[
                    "confidence"
                ],

                vader_sentiment=prediction[
                    "vader"
                ]["sentiment"],

                vader_compound=prediction[
                    "vader"
                ]["compound"],

                source="manual"
            )

            db_saved = True

        except Exception as error:

            db_warning = (
                "Failed to save prediction "
                f"to MySQL: {error}"
            )

    else:

        db_warning = (
            "MySQL is not connected. "
            "Prediction was not stored."
        )

    # --------------------------------------------------------
    # Response payload
    # --------------------------------------------------------

    response_payload = {

        "success": True,

        "id": inserted_id,

        "comment": comment,

        "domain": domain,

        "sentiment": prediction[
            "sentiment"
        ],

        "confidence": prediction[
            "confidence"
        ],

        "preprocessed_comment": prediction[
            "preprocessed_comment"
        ],

        "vader": prediction[
            "vader"
        ],

        "db_saved": db_saved,

        "db_warning": db_warning
    }

    # --------------------------------------------------------
    # JSON response
    # --------------------------------------------------------

    if request.is_json:

        return jsonify(
            response_payload
        )

    # --------------------------------------------------------
    # Browser form response
    # --------------------------------------------------------

    flash(
        (
            "Sentiment: "
            f"{prediction['sentiment'].upper()} "
            "("
            f"{int(prediction['confidence'] * 100)}%"
            " confidence)"
        ),
        "success"
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# BULK CSV PAGE
# ============================================================

@app.route("/bulk")
def bulk_page():

    status = get_db_status()

    return render_template(
        "bulk.html",
        active_page="bulk",
        db_status=status
    )


# ============================================================
# BULK CSV PREDICTION
# ============================================================

@app.route(
    "/predict-bulk",
    methods=["POST"]
)
def predict_bulk():

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if "file" not in request.files:

        flash(
            "No file part in the upload request.",
            "danger"
        )

        return redirect(
            url_for("bulk_page")
        )

    file = request.files["file"]

    # --------------------------------------------------------
    # Empty filename
    # --------------------------------------------------------

    if file.filename == "":

        flash(
            "Please select a valid CSV file.",
            "danger"
        )

        return redirect(
            url_for("bulk_page")
        )

    # --------------------------------------------------------
    # File extension
    # --------------------------------------------------------

    if not file.filename.lower().endswith(
        ".csv"
    ):

        flash(
            "Only CSV files are supported.",
            "danger"
        )

        return redirect(
            url_for("bulk_page")
        )

    try:

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        df = pd.read_csv(file)

        # ----------------------------------------------------
        # Required column
        # ----------------------------------------------------

        if "comment" not in df.columns:

            flash(
                (
                    "Uploaded CSV must contain "
                    "a column named 'comment'."
                ),
                "danger"
            )

            return redirect(
                url_for("bulk_page")
            )

        # ----------------------------------------------------
        # Remove empty comments
        # ----------------------------------------------------

        df = df.dropna(
            subset=["comment"]
        )

        # ----------------------------------------------------
        # Add default domain
        # ----------------------------------------------------

        if "domain" not in df.columns:

            df["domain"] = "general"

        db_records = []
        batch_rows = []

        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        # ----------------------------------------------------
        # Process every comment
        # ----------------------------------------------------

        for _, row in df.iterrows():

            comment_text = str(
                row["comment"]
            ).strip()

            domain = str(
                row["domain"]
            ).strip().lower()

            if not comment_text:

                continue

            prediction = (
                predict_sentiment_single(
                    comment_text
                )
            )

            sentiment = prediction[
                "sentiment"
            ]

            sentiment_counts[
                sentiment
            ] = sentiment_counts.get(
                sentiment,
                0
            ) + 1

            # ------------------------------------------------
            # Result for dashboard
            # ------------------------------------------------

            record = {

                "comment_text":
                    comment_text,

                "domain":
                    domain,

                "sentiment":
                    sentiment,

                "confidence":
                    prediction[
                        "confidence"
                    ],

                "vader_sentiment":
                    prediction[
                        "vader"
                    ]["sentiment"],

                "vader_compound":
                    prediction[
                        "vader"
                    ]["compound"],

                "source":
                    "bulk_upload"
            }

            batch_rows.append(
                record
            )

            # ------------------------------------------------
            # MySQL record
            # ------------------------------------------------

            db_records.append(
                (
                    comment_text,
                    domain,
                    sentiment,
                    prediction[
                        "confidence"
                    ],
                    prediction[
                        "vader"
                    ]["sentiment"],
                    prediction[
                        "vader"
                    ]["compound"],
                    "bulk_upload"
                )
            )

        # ----------------------------------------------------
        # SAVE BULK DATA
        # ----------------------------------------------------

        status = get_db_status()

        if (
            status["connected"]
            and db_records
        ):

            try:

                db.insert_bulk_comments(
                    db_records
                )

                flash(
                    (
                        "Successfully processed "
                        "and stored "
                        f"{len(db_records)} "
                        "comments in MySQL."
                    ),
                    "success"
                )

            except Exception as error:

                flash(
                    (
                        "Comments were processed, "
                        "but MySQL saving failed: "
                        f"{error}"
                    ),
                    "warning"
                )

        else:

            flash(
                (
                    "Processed comments, but MySQL "
                    "was unreachable. Records were "
                    "not stored in the database."
                ),
                "warning"
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        total = len(
            batch_rows
        )

        bulk_summary = {

            "total":
                total,

            "positive":
                sentiment_counts[
                    "positive"
                ],

            "negative":
                sentiment_counts[
                    "negative"
                ],

            "neutral":
                sentiment_counts[
                    "neutral"
                ],

            "positive_pct":
                round(
                    (
                        sentiment_counts[
                            "positive"
                        ]
                        / total
                        * 100
                    ),
                    1
                )
                if total > 0
                else 0,

            "negative_pct":
                round(
                    (
                        sentiment_counts[
                            "negative"
                        ]
                        / total
                        * 100
                    ),
                    1
                )
                if total > 0
                else 0,

            "neutral_pct":
                round(
                    (
                        sentiment_counts[
                            "neutral"
                        ]
                        / total
                        * 100
                    ),
                    1
                )
                if total > 0
                else 0,

            "rows":
                batch_rows
        }

        return render_template(
            "bulk.html",
            active_page="bulk",
            db_status=status,
            bulk_summary=bulk_summary
        )

    except Exception as error:

        flash(
            "Error processing CSV file: "
            f"{error}",
            "danger"
        )

        return redirect(
            url_for("bulk_page")
        )


# ============================================================
# LOAD SAMPLE DATASET
# ============================================================

@app.route("/load-sample")
def load_sample_csv():

    sample_path = os.path.join(
        BASE_DIR,
        "test_comments.csv"
    )

    # --------------------------------------------------------
    # Check sample dataset
    # --------------------------------------------------------

    if not os.path.exists(
        sample_path
    ):

        flash(
            "test_comments.csv not found.",
            "danger"
        )

        return redirect(
            url_for("bulk_page")
        )

    try:

        df = pd.read_csv(
            sample_path
        )

        # ----------------------------------------------------
        # Required column
        # ----------------------------------------------------

        if "comment" not in df.columns:

            flash(
                (
                    "test_comments.csv must "
                    "contain a 'comment' column."
                ),
                "danger"
            )

            return redirect(
                url_for("bulk_page")
            )

        db_records = []
        batch_rows = []

        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        # ----------------------------------------------------
        # Process sample dataset
        # ----------------------------------------------------

        for _, row in df.iterrows():

            comment_text = str(
                row["comment"]
            ).strip()

            domain = str(
                row.get(
                    "domain",
                    "general"
                )
            ).strip().lower()

            if not comment_text:

                continue

            prediction = (
                predict_sentiment_single(
                    comment_text
                )
            )

            sentiment = prediction[
                "sentiment"
            ]

            sentiment_counts[
                sentiment
            ] = sentiment_counts.get(
                sentiment,
                0
            ) + 1

            # ------------------------------------------------
            # Result record
            # ------------------------------------------------

            record = {

                "comment_text":
                    comment_text,

                "domain":
                    domain,

                "sentiment":
                    sentiment,

                "confidence":
                    prediction[
                        "confidence"
                    ],

                "vader_sentiment":
                    prediction[
                        "vader"
                    ]["sentiment"],

                "vader_compound":
                    prediction[
                        "vader"
                    ]["compound"],

                "source":
                    "sample_test"
            }

            batch_rows.append(
                record
            )

            # ------------------------------------------------
            # Database record
            # ------------------------------------------------

            db_records.append(
                (
                    comment_text,
                    domain,
                    sentiment,
                    prediction[
                        "confidence"
                    ],
                    prediction[
                        "vader"
                    ]["sentiment"],
                    prediction[
                        "vader"
                    ]["compound"],
                    "sample_test"
                )
            )

        # ----------------------------------------------------
        # Save to MySQL
        # ----------------------------------------------------

        status = get_db_status()

        if status["connected"]:

            try:

                if db_records:

                    db.insert_bulk_comments(
                        db_records
                    )

                flash(
                    (
                        "Sample dataset analyzed "
                        f"and {len(db_records)} "
                        "comments logged to MySQL."
                    ),
                    "success"
                )

            except Exception as error:

                flash(
                    (
                        "Sample comments analyzed, "
                        "but MySQL saving failed: "
                        f"{error}"
                    ),
                    "warning"
                )

        else:

            flash(
                (
                    "Sample dataset analyzed, "
                    "but MySQL is disconnected. "
                    "Records were not stored."
                ),
                "warning"
            )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        total = len(
            batch_rows
        )

        bulk_summary = {

            "total":
                total,

            "positive":
                sentiment_counts[
                    "positive"
                ],

            "negative":
                sentiment_counts[
                    "negative"
                ],

            "neutral":
                sentiment_counts[
                    "neutral"
                ],

            "positive_pct":
                round(
                    (
                        sentiment_counts[
                            "positive"
                        ]
                        / total
                        * 100
                    ),
                    1
                )
                if total > 0
                else 0,

            "negative_pct":
                round(
                    (
                        sentiment_counts[
                            "negative"
                        ]
                        / total
                        * 100
                    ),
                    1
                )
                if total > 0
                else 0,

            "neutral_pct":
                round(
                    (
                        sentiment_counts[
                            "neutral"
                        ]
                        / total
                        * 100
                    ),
                    1
                )
                if total > 0
                else 0,

            "rows":
                batch_rows
        }

        return render_template(
            "bulk.html",
            active_page="bulk",
            db_status=status,
            bulk_summary=bulk_summary
        )

    except Exception as error:

        flash(
            "Error loading sample dataset: "
            f"{error}",
            "danger"
        )

        return redirect(
            url_for("bulk_page")
        )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history_page():

    status = get_db_status()

    # --------------------------------------------------------
    # MySQL unavailable
    # --------------------------------------------------------

    if not status["connected"]:

        return render_template(
            "db_error.html",
            active_page="history",
            error_message=status[
                "message"
            ]
        )

    # --------------------------------------------------------
    # Search/filter
    # --------------------------------------------------------

    search = request.args.get(
        "search",
        ""
    ).strip()

    sentiment = request.args.get(
        "sentiment",
        "all"
    ).strip().lower()

    try:

        comments = db.get_all_comments(

            limit=200,

            search_keyword=(
                search
                if search
                else None
            ),

            sentiment_filter=(
                sentiment
                if sentiment != "all"
                else None
            )
        )

        return render_template(
            "history.html",

            active_page="history",

            db_status=status,

            comments=comments,

            search_keyword=search,

            current_sentiment=sentiment
        )

    except Exception as error:

        return render_template(
            "db_error.html",
            active_page="history",
            error_message=str(error)
        )


# ============================================================
# DELETE HISTORY RECORD
# ============================================================

@app.route(
    "/history/delete/<int:record_id>",
    methods=["POST"]
)
def delete_record(record_id):

    status = get_db_status()

    # --------------------------------------------------------
    # MySQL unavailable
    # --------------------------------------------------------

    if not status["connected"]:

        flash(
            "Cannot delete record: "
            "MySQL is disconnected.",
            "danger"
        )

        return redirect(
            url_for("history_page")
        )

    try:

        success = db.delete_comment_by_id(
            record_id
        )

        if success:

            flash(
                (
                    f"Comment #{record_id} "
                    "deleted successfully."
                ),
                "success"
            )

        else:

            flash(
                (
                    f"Comment #{record_id} "
                    "was not found."
                ),
                "warning"
            )

    except Exception as error:

        flash(
            "Error deleting comment: "
            f"{error}",
            "danger"
        )

    return redirect(
        url_for("history_page")
    )


# ============================================================
# API STATISTICS
# ============================================================

@app.route("/api/stats")
def api_stats():

    status = get_db_status()

    # --------------------------------------------------------
    # Database unavailable
    # --------------------------------------------------------

    if not status["connected"]:

        return jsonify(
            {
                "connected": False,
                "total": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "domain_distribution": []
            }
        )

    try:

        stats = (
            db.get_dashboard_stats()
        )

        stats["connected"] = True

        return jsonify(
            stats
        )

    except Exception as error:

        return jsonify(
            {
                "connected": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# EXPORT CSV
# ============================================================

@app.route("/export")
def export_csv():

    status = get_db_status()

    # --------------------------------------------------------
    # Database unavailable
    # --------------------------------------------------------

    if not status["connected"]:

        flash(
            "Cannot export: "
            "MySQL is not connected.",
            "danger"
        )

        return redirect(
            url_for("history_page")
        )

    try:

        comments = db.get_all_comments(
            limit=5000
        )

        df = pd.DataFrame(
            comments
        )

        output = io.StringIO()

        df.to_csv(
            output,
            index=False
        )

        output.seek(0)

        return Response(

            output.getvalue(),

            mimetype="text/csv",

            headers={
                "Content-Disposition":
                    (
                        "attachment;"
                        "filename="
                        "econsult_sentiment_records.csv"
                    )
            }
        )

    except Exception as error:

        flash(
            "Export error: "
            f"{error}",
            "danger"
        )

        return redirect(
            url_for("history_page")
        )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "base.html",
        active_page=""
    ), 404


@app.errorhandler(500)
def server_error(error):

    return render_template(
        "base.html",
        active_page=""
    ), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            5000
        )
    )

    print(
        "[*] Starting E-Consult Sentiment AI "
        f"on http://127.0.0.1:{port}"
    )

    app.run(
        host="127.0.0.1",
        port=port,
        debug=True
    )