"""
MySQL Database Manager for E-Consultation Sentiment Analysis.

Handles connection to MySQL database 'sentiment_analysis', manages the 'comments'
table, and provides clean CRUD methods with user-friendly error diagnostics.
Strictly requires MySQL (no silent fallback).
"""

import os
import time
import socket
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_HOST = (os.getenv('DB_HOST') or '127.0.0.1').strip()
if DB_HOST.lower() == 'localhost':
    DB_HOST = '127.0.0.1'

DB_PORT = int((os.getenv('DB_PORT') or 3306))
DB_USER = (os.getenv('DB_USER') or 'root').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD') or ''
DB_NAME = (os.getenv('DB_NAME') or 'sentiment_analysis').strip()

# Cache DB connection status for 5 seconds to prevent repeated timeouts
_STATUS_CACHE = {'connected': False, 'message': '', 'expires_at': 0}


class DatabaseConnectionError(Exception):
    """Custom exception raised when MySQL connection fails."""
    pass


def is_port_open(host, port, timeout=0.5):
    """Fast check to see if MySQL port is actively listening."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_connection(use_database=True):
    """
    Establishes a connection to MySQL server.
    Raises DatabaseConnectionError with actionable diagnosis if unavailable.
    """
    if not is_port_open(DB_HOST, DB_PORT, timeout=0.5):
        raise DatabaseConnectionError(
            f"Cannot reach MySQL server at {DB_HOST}:{DB_PORT}. "
            "Is MySQL running (e.g. via XAMPP, WAMP, or Windows Service)?"
        )

    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME if use_database else None,
            cursorclass=DictCursor,
            connect_timeout=2,
            autocommit=True,
            charset='utf8mb4'
        )
        return conn
    except pymysql.MySQLError as err:
        error_code = err.args[0] if err.args else 'UNKNOWN'
        error_msg = str(err)
        
        if error_code in (2003, 2002):
            hint = f"Cannot reach MySQL server at {DB_HOST}:{DB_PORT}. Is MySQL running?"
        elif error_code == 1045:
            hint = f"Access denied for user '{DB_USER}'@{DB_HOST}. Please verify your DB_PASSWORD in .env."
        elif error_code == 1049 and use_database:
            hint = f"Database '{DB_NAME}' does not exist yet. Run init_db() to create it automatically."
        else:
            hint = f"MySQL Error [{error_code}]: {error_msg}"
            
        raise DatabaseConnectionError(f"{error_msg} | Hint: {hint}") from err


def check_connection():
    """
    Checks if MySQL is reachable with a 5-second cache to prevent latency.
    Returns (status: bool, message: str).
    """
    global _STATUS_CACHE
    now = time.time()
    if now < _STATUS_CACHE['expires_at']:
        return _STATUS_CACHE['connected'], _STATUS_CACHE['message']

    if not is_port_open(DB_HOST, DB_PORT, timeout=0.5):
        msg = f"Cannot reach MySQL server at {DB_HOST}:{DB_PORT}. Please start MySQL (e.g., via XAMPP Control Panel)."
        _STATUS_CACHE = {'connected': False, 'message': msg, 'expires_at': now + 5}
        return False, msg

    try:
        conn = get_connection(use_database=False)
        conn.close()
        _STATUS_CACHE = {'connected': True, 'message': "MySQL Connected Successfully", 'expires_at': now + 10}
        return True, "MySQL Connected Successfully"
    except DatabaseConnectionError as e:
        _STATUS_CACHE = {'connected': False, 'message': str(e), 'expires_at': now + 5}
        return False, str(e)


def init_db():
    """
    Initializes the MySQL database and 'comments' table.
    Creates the database if it doesn't exist.
    """
    # 1. Connect without selecting database to create database
    try:
        conn = get_connection(use_database=False)
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        conn.close()
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to create database '{DB_NAME}': {e}")

    # 2. Connect to the database and create table
    try:
        conn = get_connection(use_database=True)
        with conn.cursor() as cursor:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS comments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                comment_text TEXT NOT NULL,
                domain VARCHAR(100) DEFAULT 'general',
                sentiment VARCHAR(20) NOT NULL,
                confidence FLOAT NOT NULL,
                vader_sentiment VARCHAR(20) DEFAULT NULL,
                vader_compound FLOAT DEFAULT NULL,
                source VARCHAR(50) DEFAULT 'manual',
                demo_session_id VARCHAR(100) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_demo_session (demo_session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_sql)

            # Safe conditional migration: add demo_session_id column if not present
            cursor.execute("SHOW COLUMNS FROM comments LIKE 'demo_session_id'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE comments ADD COLUMN demo_session_id VARCHAR(100) NULL")
                try:
                    cursor.execute("CREATE INDEX idx_demo_session ON comments (demo_session_id)")
                except Exception:
                    pass
        conn.close()
        return True
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to initialize 'comments' table: {e}")


def insert_comment(comment_text, domain, sentiment, confidence,
                   vader_sentiment=None, vader_compound=None, source='manual',
                   demo_session_id=None):
    """Inserts a single prediction record into MySQL with optional session identifier."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO comments 
            (comment_text, domain, sentiment, confidence, vader_sentiment, vader_compound, source, demo_session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                comment_text, domain, sentiment, confidence,
                vader_sentiment, vader_compound, source, demo_session_id
            ))
            return cursor.lastrowid
    finally:
        conn.close()


def insert_bulk_comments(records):
    """Inserts multiple comment records in batch."""
    if not records:
        return 0
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            if len(records[0]) == 8:
                sql = """
                INSERT INTO comments 
                (comment_text, domain, sentiment, confidence, vader_sentiment, vader_compound, source, demo_session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
            else:
                sql = """
                INSERT INTO comments 
                (comment_text, domain, sentiment, confidence, vader_sentiment, vader_compound, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
            cursor.executemany(sql, records)
            return cursor.rowcount
    finally:
        conn.close()


def get_distinct_domains():
    """Returns a sorted list of unique domain strings from the comments table."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT domain FROM comments "
                "WHERE domain IS NOT NULL AND domain != '' "
                "ORDER BY domain ASC"
            )
            rows = cursor.fetchall()
            return [r['domain'] for r in rows if r.get('domain')]
    finally:
        conn.close()


def get_all_comments(
    limit=25,
    offset=0,
    search_keyword=None,
    sentiment_filter=None,
    domain_filter=None,
    date_from=None,
    date_to=None,
    session_id=None,
    return_total=False
):
    """
    Fetches stored comments with optional keyword, sentiment, domain, date, and session filtering.
    Supports pagination via limit & offset, and optionally returns (records, total_count).
    """
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            where_conditions = ["1=1"]
            params = []
            
            if session_id:
                where_conditions.append("demo_session_id = %s")
                params.append(session_id)
                
            if search_keyword:
                where_conditions.append("comment_text LIKE %s")
                params.append(f"%{search_keyword}%")
                
            if sentiment_filter and sentiment_filter.lower() != 'all':
                where_conditions.append("sentiment = %s")
                params.append(sentiment_filter.lower())
                
            if domain_filter and domain_filter.lower() != 'all':
                where_conditions.append("domain = %s")
                params.append(domain_filter)

            if date_from:
                where_conditions.append("created_at >= %s")
                params.append(f"{date_from} 00:00:00")

            if date_to:
                where_conditions.append("created_at <= %s")
                params.append(f"{date_to} 23:59:59")

            where_sql = " WHERE " + " AND ".join(where_conditions)

            total_count = 0
            if return_total:
                count_query = f"SELECT COUNT(*) AS total FROM comments{where_sql}"
                cursor.execute(count_query, params)
                count_row = cursor.fetchone()
                total_count = count_row['total'] if count_row else 0

            data_query = f"SELECT * FROM comments{where_sql} ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
            data_params = list(params) + [int(limit), int(offset)]
            cursor.execute(data_query, data_params)
            records = cursor.fetchall()

            if return_total:
                return records, total_count
            return records
    finally:
        conn.close()


def get_dashboard_stats(session_id=None):
    """Calculates summary KPIs and domain-wise metrics from MySQL, optionally scoped to a session."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            where_clause = " WHERE demo_session_id = %s" if session_id else ""
            params = (session_id,) if session_id else ()
            
            cursor.execute(f"SELECT COUNT(*) AS total FROM comments{where_clause}", params)
            total = cursor.fetchone()['total']
            
            cursor.execute(
                f"SELECT sentiment, COUNT(*) AS count FROM comments{where_clause} GROUP BY sentiment",
                params
            )
            sentiment_rows = cursor.fetchall()
            
            cursor.execute(
                f"SELECT domain, sentiment, COUNT(*) AS count FROM comments{where_clause} GROUP BY domain, sentiment",
                params
            )
            domain_rows = cursor.fetchall()
            
            cursor.execute(
                f"SELECT * FROM comments{where_clause} ORDER BY created_at DESC LIMIT 10",
                params
            )
            recent_comments = cursor.fetchall()
            
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for row in sentiment_rows:
            s = row['sentiment'].lower()
            if s in sentiment_counts:
                sentiment_counts[s] = row['count']
                
        pos = sentiment_counts['positive']
        neg = sentiment_counts['negative']
        neu = sentiment_counts['neutral']

        # Structured domain-wise sentiment distribution
        # [{"domain": "general", "positive": 2, "negative": 1, "neutral": 0, "total": 3}, ...]
        domain_map = {}
        for row in domain_rows:
            dom = (row['domain'] or 'general').strip()
            if dom not in domain_map:
                domain_map[dom] = {
                    'domain': dom,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'total': 0
                }
            s = (row['sentiment'] or '').strip().lower()
            cnt = int(row['count'])
            if s in ('positive', 'negative', 'neutral'):
                domain_map[dom][s] += cnt
                domain_map[dom]['total'] += cnt

        domain_distribution = sorted(
            domain_map.values(),
            key=lambda x: (-x['total'], x['domain'])
        )

        stats = {
            'total': total,
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'positive_pct': round((pos / total * 100), 1) if total > 0 else 0,
            'negative_pct': round((neg / total * 100), 1) if total > 0 else 0,
            'neutral_pct': round((neu / total * 100), 1) if total > 0 else 0,
            'domain_distribution': domain_distribution,
            'recent_comments': recent_comments,
            'session_id': session_id
        }
        return stats
    finally:
        conn.close()


def delete_comment_by_id(comment_id):
    """Deletes a comment by ID."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM comments WHERE id = %s", (comment_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================================
# ADVANCED ANALYTICS (EXTENSION #8)
# ============================================================

def _build_analytics_where(session_id=None, date_from=None, date_to=None):
    """Constructs parameterized WHERE conditions and parameters for analytical queries."""
    where_clauses = ["1=1"]
    params = []
    if session_id:
        where_clauses.append("demo_session_id = %s")
        params.append(session_id)
    if date_from:
        where_clauses.append("created_at >= %s")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        where_clauses.append("created_at <= %s")
        params.append(f"{date_to} 23:59:59")
    return " WHERE " + " AND ".join(where_clauses), params


def get_analytics_summary(session_id=None, date_from=None, date_to=None):
    """
    Returns high-level summary KPIs for the advanced analytics view:
    total comments, distinct domains count, positive/negative/neutral counts and percentages,
    and average ML confidence score.
    """
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_analytics_where(session_id, date_from, date_to)
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) AS total, COUNT(DISTINCT domain) AS num_domains, "
                f"AVG(confidence) AS avg_confidence FROM comments{where_sql}",
                params
            )
            agg_row = cursor.fetchone() or {}
            total = int(agg_row.get('total') or 0)
            num_domains = int(agg_row.get('num_domains') or 0)
            avg_conf = float(agg_row.get('avg_confidence') or 0.0)

            cursor.execute(
                f"SELECT sentiment, COUNT(*) AS count FROM comments{where_sql} GROUP BY sentiment",
                params
            )
            sent_rows = cursor.fetchall()

        sent_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for r in sent_rows:
            s = (r.get('sentiment') or '').strip().lower()
            if s in sent_counts:
                sent_counts[s] = int(r.get('count') or 0)

        pos = sent_counts['positive']
        neg = sent_counts['negative']
        neu = sent_counts['neutral']

        return {
            'total': total,
            'num_domains': num_domains,
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'positive_pct': round((pos / total * 100), 1) if total > 0 else 0.0,
            'negative_pct': round((neg / total * 100), 1) if total > 0 else 0.0,
            'neutral_pct': round((neu / total * 100), 1) if total > 0 else 0.0,
            'avg_confidence': round(avg_conf * 100, 2) if total > 0 else 0.0,
            'session_id': session_id
        }
    finally:
        conn.close()


def get_sentiment_trend(period='daily', session_id=None, date_from=None, date_to=None):
    """
    Aggregates comments over time grouped by the chosen period:
    - 'daily': DATE(created_at) -> YYYY-MM-DD
    - 'weekly': DATE(DATE_SUB(created_at, INTERVAL WEEKDAY(created_at) DAY)) -> YYYY-MM-DD
    - 'monthly': DATE_FORMAT(created_at, '%Y-%m') -> YYYY-MM
    Returns sorted timeline labels and sentiment counts.
    """
    period = (period or 'daily').strip().lower()
    if period == 'weekly':
        date_expr = "DATE(DATE_SUB(created_at, INTERVAL WEEKDAY(created_at) DAY))"
    elif period == 'monthly':
        date_expr = "DATE_FORMAT(created_at, '%%Y-%%m')"
    else:
        date_expr = "DATE(created_at)"

    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_analytics_where(session_id, date_from, date_to)
        query = (
            f"SELECT {date_expr} AS time_bucket, sentiment, COUNT(*) AS count "
            f"FROM comments{where_sql} "
            f"GROUP BY time_bucket, sentiment "
            f"ORDER BY time_bucket ASC"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        timeline_map = {}
        for r in rows:
            raw_bucket = r.get('time_bucket')
            if not raw_bucket:
                continue
            bucket = str(raw_bucket)
            if bucket not in timeline_map:
                timeline_map[bucket] = {'positive': 0, 'negative': 0, 'neutral': 0}
            s = (r.get('sentiment') or '').strip().lower()
            if s in timeline_map[bucket]:
                timeline_map[bucket][s] += int(r.get('count') or 0)

        labels = sorted(timeline_map.keys())
        pos_data = [timeline_map[l]['positive'] for l in labels]
        neg_data = [timeline_map[l]['negative'] for l in labels]
        neu_data = [timeline_map[l]['neutral'] for l in labels]

        return {
            'period': period,
            'labels': labels,
            'positive': pos_data,
            'negative': neg_data,
            'neutral': neu_data
        }
    finally:
        conn.close()


def get_domain_sentiment_analytics(session_id=None, date_from=None, date_to=None, limit=15):
    """
    Returns domain-wise sentiment distribution (Positive, Negative, Neutral, Total)
    sorted by total comment volume descending.
    """
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_analytics_where(session_id, date_from, date_to)
        query = (
            f"SELECT domain, sentiment, COUNT(*) AS count "
            f"FROM comments{where_sql} "
            f"GROUP BY domain, sentiment"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        domain_map = {}
        for r in rows:
            dom = (r.get('domain') or 'general').strip()
            if dom not in domain_map:
                domain_map[dom] = {
                    'domain': dom,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'total': 0
                }
            s = (r.get('sentiment') or '').strip().lower()
            cnt = int(r.get('count') or 0)
            if s in ('positive', 'negative', 'neutral'):
                domain_map[dom][s] += cnt
                domain_map[dom]['total'] += cnt

        sorted_domains = sorted(domain_map.values(), key=lambda x: -x['total'])
        return sorted_domains[:int(limit)]
    finally:
        conn.close()


def get_top_feedback(sentiment='positive', limit=5, session_id=None, date_from=None, date_to=None):
    """
    Retrieves highest-confidence feedback comments for the given sentiment class.
    """
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_analytics_where(session_id, date_from, date_to)
        sentiment_val = (sentiment or 'positive').strip().lower()
        where_sql += " AND sentiment = %s"
        params.append(sentiment_val)

        query = (
            f"SELECT id, comment_text, domain, sentiment, confidence, created_at "
            f"FROM comments{where_sql} "
            f"ORDER BY confidence DESC, id DESC "
            f"LIMIT %s"
        )
        params.append(int(limit))
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            for r in rows:
                if 'created_at' in r and r['created_at']:
                    r['created_at_str'] = str(r['created_at'])
            return rows
    finally:
        conn.close()


def get_keyword_frequency(session_id=None, date_from=None, date_to=None, limit=10):
    """
    Extracts high-frequency sentiment keywords using NLTK preprocessing on
    sample comments matching the filter criteria.
    Returns {'positive': [{'word': ..., 'count': ...}], 'negative': [{'word': ..., 'count': ...}]}.
    """
    import collections
    import preprocess

    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_analytics_where(session_id, date_from, date_to)
        with conn.cursor() as cursor:
            pos_query = f"SELECT comment_text FROM comments{where_sql} AND sentiment = 'positive' LIMIT 300"
            cursor.execute(pos_query, params)
            pos_rows = cursor.fetchall()

            neg_query = f"SELECT comment_text FROM comments{where_sql} AND sentiment = 'negative' LIMIT 300"
            cursor.execute(neg_query, params)
            neg_rows = cursor.fetchall()

        stopwords_extra = {'not', 'no', 'doctor', 'patient', 'consultation', 'get', 'service', 'econsult'}

        pos_counter = collections.Counter()
        for r in pos_rows:
            txt = r.get('comment_text') or ''
            tokens = preprocess.preprocess_comment(txt).split()
            pos_counter.update(w for w in tokens if len(w) > 2 and w not in stopwords_extra)

        neg_counter = collections.Counter()
        for r in neg_rows:
            txt = r.get('comment_text') or ''
            tokens = preprocess.preprocess_comment(txt).split()
            neg_counter.update(w for w in tokens if len(w) > 2 and w not in stopwords_extra)

        return {
            'positive': [{'word': w, 'count': c} for w, c in pos_counter.most_common(int(limit))],
            'negative': [{'word': w, 'count': c} for w, c in neg_counter.most_common(int(limit))]
        }
    finally:
        conn.close()


# ============================================================
# FEEDBACK INSIGHTS (EXTENSION #9)
# ============================================================

def _build_insights_where(session_id=None, date_from=None, date_to=None, domain=None, sentiment=None):
    """Constructs parameterized WHERE conditions and parameters for Feedback Insights queries."""
    where_clauses = ["1=1"]
    params = []
    if session_id:
        where_clauses.append("demo_session_id = %s")
        params.append(session_id)
    if date_from:
        where_clauses.append("created_at >= %s")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        where_clauses.append("created_at <= %s")
        params.append(f"{date_to} 23:59:59")
    if domain and domain.strip() and domain.strip().lower() != 'all':
        where_clauses.append("domain = %s")
        params.append(domain.strip())
    if sentiment and sentiment.strip() and sentiment.strip().lower() in {'positive', 'negative', 'neutral'}:
        where_clauses.append("sentiment = %s")
        params.append(sentiment.strip().lower())
    return " WHERE " + " AND ".join(where_clauses), params


def get_insights_summary(session_id=None, date_from=None, date_to=None, domain=None, sentiment=None):
    """
    Computes top-level KPI summary for Feedback Insights:
    - total, positive, negative, neutral counts and percentages
    - average confidence
    - most active domain (name and count)
    """
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_insights_where(session_id, date_from, date_to, domain, sentiment)
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) AS total, AVG(confidence) AS avg_confidence FROM comments{where_sql}",
                params
            )
            agg_row = cursor.fetchone() or {}
            total = int(agg_row.get('total') or 0)
            avg_conf = float(agg_row.get('avg_confidence') or 0.0)

            cursor.execute(
                f"SELECT sentiment, COUNT(*) AS count FROM comments{where_sql} GROUP BY sentiment",
                params
            )
            sent_rows = cursor.fetchall()

            cursor.execute(
                f"SELECT domain, COUNT(*) AS count FROM comments{where_sql} "
                f"AND domain IS NOT NULL AND domain != '' "
                f"GROUP BY domain ORDER BY count DESC LIMIT 1",
                params
            )
            active_dom_row = cursor.fetchone() or {}

        sent_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for r in sent_rows:
            s = (r.get('sentiment') or '').strip().lower()
            if s in sent_counts:
                sent_counts[s] = int(r.get('count') or 0)

        pos = sent_counts['positive']
        neg = sent_counts['negative']
        neu = sent_counts['neutral']

        most_active_domain = active_dom_row.get('domain') or 'None'
        most_active_domain_count = int(active_dom_row.get('count') or 0)

        return {
            'total': total,
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'positive_pct': round((pos / total * 100), 1) if total > 0 else 0.0,
            'negative_pct': round((neg / total * 100), 1) if total > 0 else 0.0,
            'neutral_pct': round((neu / total * 100), 1) if total > 0 else 0.0,
            'average_confidence': round(avg_conf * 100, 2) if total > 0 else 0.0,
            'most_active_domain': most_active_domain,
            'most_active_domain_count': most_active_domain_count,
            'session_id': session_id
        }
    finally:
        conn.close()


def get_top_positive_feedback(limit=5, session_id=None, date_from=None, date_to=None, domain=None):
    """Retrieves highest-confidence positive feedback comments."""
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_insights_where(session_id, date_from, date_to, domain, sentiment='positive')
        query = (
            f"SELECT id, comment_text, domain, sentiment, confidence, created_at "
            f"FROM comments{where_sql} "
            f"ORDER BY confidence DESC, id DESC "
            f"LIMIT %s"
        )
        params.append(int(limit))
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            for r in rows:
                if 'created_at' in r and r['created_at']:
                    r['created_at_str'] = str(r['created_at'])
            return rows
    finally:
        conn.close()


def get_top_negative_feedback(limit=5, session_id=None, date_from=None, date_to=None, domain=None):
    """Retrieves highest-confidence negative feedback comments."""
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_insights_where(session_id, date_from, date_to, domain, sentiment='negative')
        query = (
            f"SELECT id, comment_text, domain, sentiment, confidence, created_at "
            f"FROM comments{where_sql} "
            f"ORDER BY confidence DESC, id DESC "
            f"LIMIT %s"
        )
        params.append(int(limit))
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            for r in rows:
                if 'created_at' in r and r['created_at']:
                    r['created_at_str'] = str(r['created_at'])
            return rows
    finally:
        conn.close()


def get_nlp_keywords_and_frequent_words(session_id=None, date_from=None, date_to=None, domain=None, sentiment=None, limit=10):
    """
    Performs NLP tokenization, stopword removal, and lemmatization using preprocess.py.
    Extracts:
    - frequent_words: overall top N meaningful words across matched comments
    - positive_keywords: top N meaningful words across positive comments
    - negative_keywords: top N meaningful words across negative comments
    """
    import collections
    import preprocess

    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_insights_where(session_id, date_from, date_to, domain, sentiment)
        query = f"SELECT comment_text, sentiment FROM comments{where_sql} LIMIT 600"
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        filter_stops = {
            'not', 'no', 'neither', 'nor', 'never', 'cannot', 'would', 'could', 'get',
            'also', 'one', 'take', 'make', 'need', 'go', 'use', 'see', 'give', 'much',
            'well', 'way', 'even', 'many', 'still', 'must', 'come', 'like'
        }

        all_counter = collections.Counter()
        pos_counter = collections.Counter()
        neg_counter = collections.Counter()

        for r in rows:
            txt = r.get('comment_text') or ''
            s = (r.get('sentiment') or '').strip().lower()
            tokens = [
                w for w in preprocess.preprocess_comment(txt).split()
                if len(w) > 2 and w not in filter_stops
            ]
            all_counter.update(tokens)
            if s == 'positive':
                pos_counter.update(tokens)
            elif s == 'negative':
                neg_counter.update(tokens)

        return {
            'frequent_words': [{'word': w, 'count': c} for w, c in all_counter.most_common(int(limit))],
            'positive_keywords': [{'word': w, 'count': c} for w, c in pos_counter.most_common(int(limit))],
            'negative_keywords': [{'word': w, 'count': c} for w, c in neg_counter.most_common(int(limit))]
        }
    finally:
        conn.close()


def get_problem_categories(session_id=None, date_from=None, date_to=None, domain=None, sentiment=None):
    """
    Applies rule-based regex keyword categorization on feedback comments.
    Categories: Technical Issue, Waiting / Delay, Doctor / Consultation, Service Quality,
    Privacy / Security, Appointment, Payment, Nutrition, General Feedback.
    """
    import re
    import collections

    CATEGORY_RULES = [
        ('Technical Issue', re.compile(r'\b(crash|error|bug|website|login|system|loading|fail|failed|failure|slow|portal|network|connection|down|glitch|freeze|timeout)\b', re.IGNORECASE)),
        ('Waiting / Delay', re.compile(r'\b(wait|waiting|delay|delayed|late|queue|hold|reschedule|cancel|cancelled|postpone|hours?|slow)\b', re.IGNORECASE)),
        ('Doctor / Consultation', re.compile(r'\b(doctor|physician|consultation|specialist|diagnosis|prescription|prescribe|medical|clinician|practitioner|treatment)\b', re.IGNORECASE)),
        ('Service Quality', re.compile(r'\b(quality|staff|attitude|rude|poor|unprofessional|unhelpful|terrible|awful|horrible|great|excellent|helpful|polite|professional)\b', re.IGNORECASE)),
        ('Privacy / Security', re.compile(r'\b(privacy|security|password|confidential|leak|hack|consent|private|secure|breach)\b', re.IGNORECASE)),
        ('Appointment', re.compile(r'\b(appointment|booking|book|booked|schedule|scheduled|scheduling|slot|slots|visit)\b', re.IGNORECASE)),
        ('Payment', re.compile(r'\b(payment|fee|fees|charge|charged|refund|cost|expensive|bill|billing|price|pricing|card|transaction)\b', re.IGNORECASE)),
        ('Nutrition', re.compile(r'\b(nutrition|diet|dietary|food|meal|supplement|supplements|weight|calorie|calories|vitamin|vitamins)\b', re.IGNORECASE))
    ]

    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_insights_where(session_id, date_from, date_to, domain, sentiment)
        query = f"SELECT comment_text, sentiment FROM comments{where_sql}"
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        total = len(rows)
        cat_counts = collections.Counter()
        neg_cat_counts = collections.Counter()

        for r in rows:
            txt = r.get('comment_text') or ''
            s = (r.get('sentiment') or '').strip().lower()
            matched = False
            for cat_name, pattern in CATEGORY_RULES:
                if pattern.search(txt):
                    cat_counts[cat_name] += 1
                    if s == 'negative':
                        neg_cat_counts[cat_name] += 1
                    matched = True
                    break
            if not matched:
                cat_counts['General Feedback'] += 1
                if s == 'negative':
                    neg_cat_counts['General Feedback'] += 1

        categories = []
        for cat_name, _ in CATEGORY_RULES:
            cnt = cat_counts[cat_name]
            neg_cnt = neg_cat_counts[cat_name]
            pct = round((cnt / total * 100), 1) if total > 0 else 0.0
            categories.append({
                'name': cat_name,
                'count': cnt,
                'percentage': pct,
                'negative_count': neg_cnt
            })

        gen_cnt = cat_counts['General Feedback']
        categories.append({
            'name': 'General Feedback',
            'count': gen_cnt,
            'percentage': round((gen_cnt / total * 100), 1) if total > 0 else 0.0,
            'negative_count': neg_cat_counts['General Feedback']
        })

        sorted_cats = sorted(categories, key=lambda x: -x['count'])
        return {
            'total_analyzed': total,
            'categories': sorted_cats
        }
    finally:
        conn.close()


def get_domain_insights(session_id=None, date_from=None, date_to=None, sentiment=None, limit=20):
    """
    Computes domain-specific metrics: total feedback, positive/negative/neutral counts,
    positive/negative percentages, and average ML confidence.
    """
    conn = get_connection(use_database=True)
    try:
        where_sql, params = _build_insights_where(session_id, date_from, date_to, domain=None, sentiment=sentiment)
        query = (
            f"SELECT domain, sentiment, COUNT(*) AS count, AVG(confidence) AS avg_conf "
            f"FROM comments{where_sql} "
            f"AND domain IS NOT NULL AND domain != '' "
            f"GROUP BY domain, sentiment"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        domain_map = {}
        for r in rows:
            dom = (r.get('domain') or 'general').strip()
            if dom not in domain_map:
                domain_map[dom] = {
                    'domain': dom,
                    'total': 0,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'conf_sum': 0.0,
                    'conf_count': 0
                }
            s = (r.get('sentiment') or '').strip().lower()
            cnt = int(r.get('count') or 0)
            conf = float(r.get('avg_conf') or 0.0)
            if s in ('positive', 'negative', 'neutral'):
                domain_map[dom][s] += cnt
                domain_map[dom]['total'] += cnt
                domain_map[dom]['conf_sum'] += conf * cnt
                domain_map[dom]['conf_count'] += cnt

        result = []
        for dom, data in domain_map.items():
            tot = data['total']
            pos = data['positive']
            neg = data['negative']
            neu = data['neutral']
            avg_c = (data['conf_sum'] / data['conf_count']) if data['conf_count'] > 0 else 0.0
            result.append({
                'domain': dom,
                'total': tot,
                'positive': pos,
                'negative': neg,
                'neutral': neu,
                'positive_pct': round((pos / tot * 100), 1) if tot > 0 else 0.0,
                'negative_pct': round((neg / tot * 100), 1) if tot > 0 else 0.0,
                'neutral_pct': round((neu / tot * 100), 1) if tot > 0 else 0.0,
                'average_confidence': round(avg_c * 100, 2)
            })

        sorted_result = sorted(result, key=lambda x: -x['total'])
        return sorted_result[:int(limit)]
    finally:
        conn.close()


def generate_actionable_insights(summary, domain_insights, problem_categories):
    """
    Synthesizes factual, data-grounded administrative summary statements from computed metrics.
    No unsupported claims, subjective language, or hallucinated values.
    """
    insights = []
    total = summary.get('total', 0)
    if total == 0:
        return ["No feedback records found matching the active filter criteria."]

    pos_pct = summary.get('positive_pct', 0.0)
    neg_pct = summary.get('negative_pct', 0.0)
    neu_pct = summary.get('neutral_pct', 0.0)
    avg_conf = summary.get('average_confidence', 0.0)
    most_act_dom = summary.get('most_active_domain', 'None')
    most_act_cnt = summary.get('most_active_domain_count', 0)

    insights.append(
        f"Analyzed {total:,} total e-consultation feedback records across {len(domain_insights)} distinct consultation domains."
    )
    insights.append(
        f"Overall sentiment distribution stands at {pos_pct}% positive ({summary.get('positive', 0):,} records), "
        f"{neg_pct}% negative ({summary.get('negative', 0):,} records), and {neu_pct}% neutral ({summary.get('neutral', 0):,} records)."
    )

    if most_act_dom != 'None' and most_act_cnt > 0:
        dom_share = round((most_act_cnt / total * 100), 1) if total > 0 else 0.0
        insights.append(
            f"The consultation domain with the highest engagement is '{most_act_dom}', accounting for {most_act_cnt:,} comments ({dom_share}% of total feedback)."
        )

    if domain_insights:
        top_pos = max(domain_insights, key=lambda d: d.get('positive', 0), default=None)
        if top_pos and top_pos.get('positive', 0) > 0:
            insights.append(
                f"Domain '{top_pos['domain']}' recorded the highest positive feedback volume with {top_pos['positive']:,} positive comments ({top_pos['positive_pct']}% positive rate)."
            )

        top_neg = max(domain_insights, key=lambda d: d.get('negative', 0), default=None)
        if top_neg and top_neg.get('negative', 0) > 0:
            insights.append(
                f"Domain '{top_neg['domain']}' recorded the highest negative feedback volume with {top_neg['negative']:,} negative comments ({top_neg['negative_pct']}% negative rate)."
            )

    cats = problem_categories.get('categories', [])
    specific_cats = [c for c in cats if c['name'] != 'General Feedback' and c['count'] > 0]
    if specific_cats:
        top_cat = specific_cats[0]
        insights.append(
            f"The primary specific administrative category identified is '{top_cat['name']}' with {top_cat['count']:,} comments ({top_cat['percentage']}% of analyzed records)."
        )

    insights.append(
        f"The machine learning sentiment classification pipeline maintained an average prediction confidence of {avg_conf}% across the active dataset."
    )

    return insights
