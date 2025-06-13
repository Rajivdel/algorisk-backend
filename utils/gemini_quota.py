import sqlite3
from datetime import datetime

DB_NAME = "users.db"  # Use your existing DB

MONTHLY_TOKEN_LIMIT = 1000000  # Set your monthly Gemini token limit

def init_gemini_token_usage_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS gemini_token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def get_current_month():
    return datetime.now().strftime("%Y-%m")

def get_tokens_used_this_month():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT tokens_used FROM gemini_token_usage WHERE month = ?", (get_current_month(),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def add_tokens_used(num_tokens):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    month = get_current_month()
    c.execute("SELECT tokens_used FROM gemini_token_usage WHERE month = ?", (month,))
    row = c.fetchone()
    if row:
        c.execute("UPDATE gemini_token_usage SET tokens_used = tokens_used + ? WHERE month = ?", (num_tokens, month))
    else:
        c.execute("INSERT INTO gemini_token_usage (month, tokens_used) VALUES (?, ?)", (month, num_tokens))
    conn.commit()
    conn.close()

def can_use_gemini(num_tokens_needed):
    tokens_used = get_tokens_used_this_month()
    return tokens_used + num_tokens_needed <= MONTHLY_TOKEN_LIMIT
