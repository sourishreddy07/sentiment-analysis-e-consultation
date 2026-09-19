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

DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
if DB_HOST.lower() == 'localhost':
    DB_HOST = '127.0.0.1'

DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'sentiment_analysis')

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_sql)
        conn.close()
        return True
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to initialize 'comments' table: {e}")


def insert_comment(comment_text, domain, sentiment, confidence,
                   vader_sentiment=None, vader_compound=None, source='manual'):
    """Inserts a single prediction record into MySQL."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO comments 
            (comment_text, domain, sentiment, confidence, vader_sentiment, vader_compound, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                comment_text, domain, sentiment, confidence,
                vader_sentiment, vader_compound, source
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
            sql = """
            INSERT INTO comments 
            (comment_text, domain, sentiment, confidence, vader_sentiment, vader_compound, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.executemany(sql, records)
            return cursor.rowcount
    finally:
        conn.close()


def get_all_comments(limit=200, search_keyword=None, sentiment_filter=None):
    """Fetches stored comments with optional keyword and sentiment filtering."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            query = "SELECT * FROM comments WHERE 1=1"
            params = []
            
            if search_keyword:
                query += " AND comment_text LIKE %s"
                params.append(f"%{search_keyword}%")
                
            if sentiment_filter and sentiment_filter.lower() != 'all':
                query += " AND sentiment = %s"
                params.append(sentiment_filter.lower())
                
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(int(limit))
            
            cursor.execute(query, params)
            return cursor.fetchall()
    finally:
        conn.close()


def get_dashboard_stats():
    """Calculates summary KPIs and domain-wise metrics from MySQL."""
    conn = get_connection(use_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM comments")
            total = cursor.fetchone()['total']
            
            cursor.execute(
                "SELECT sentiment, COUNT(*) AS count FROM comments GROUP BY sentiment"
            )
            sentiment_rows = cursor.fetchall()
            
            cursor.execute(
                "SELECT domain, sentiment, COUNT(*) AS count FROM comments GROUP BY domain, sentiment"
            )
            domain_rows = cursor.fetchall()
            
            cursor.execute(
                "SELECT * FROM comments ORDER BY created_at DESC LIMIT 10"
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
        
        stats = {
            'total': total,
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'positive_pct': round((pos / total * 100), 1) if total > 0 else 0,
            'negative_pct': round((neg / total * 100), 1) if total > 0 else 0,
            'neutral_pct': round((neu / total * 100), 1) if total > 0 else 0,
            'domain_distribution': domain_rows,
            'recent_comments': recent_comments
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
