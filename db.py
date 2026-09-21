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
