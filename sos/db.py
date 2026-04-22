"""
SOS system database helper — all SQLite read/write lives here.
DB path: ~/sos_system.db
"""

import os
import sqlite3

DB_PATH = os.path.expanduser('~/sos_system.db')


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they don't exist."""
    with _conn() as c:
        c.executescript('''
            CREATE TABLE IF NOT EXISTS user (
                id           INTEGER PRIMARY KEY,
                name         TEXT NOT NULL,
                blood_group  TEXT,
                medical_notes TEXT
            );
            CREATE TABLE IF NOT EXISTS contacts (
                id              INTEGER PRIMARY KEY,
                contact_name    TEXT NOT NULL,
                telegram_chat_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bot_config (
                id        INTEGER PRIMARY KEY,
                bot_token TEXT NOT NULL
            );
        ''')


# ── Writers ────────────────────────────────────────────────────────────────

def save_user(name, blood_group, medical_notes):
    with _conn() as c:
        c.execute('DELETE FROM user')
        c.execute('INSERT INTO user (name, blood_group, medical_notes) VALUES (?,?,?)',
                  (name, blood_group, medical_notes))


def save_contact(contact_name, telegram_chat_id):
    with _conn() as c:
        c.execute('DELETE FROM contacts')
        c.execute('INSERT INTO contacts (contact_name, telegram_chat_id) VALUES (?,?)',
                  (contact_name, str(telegram_chat_id)))


def save_bot_token(token):
    with _conn() as c:
        c.execute('DELETE FROM bot_config')
        c.execute('INSERT INTO bot_config (bot_token) VALUES (?)', (token,))


# ── Readers ────────────────────────────────────────────────────────────────

def load_user():
    """Returns (name, blood_group, medical_notes) or None."""
    with _conn() as c:
        return c.execute(
            'SELECT name, blood_group, medical_notes FROM user LIMIT 1'
        ).fetchone()


def load_contact():
    """Returns (contact_name, telegram_chat_id) or None."""
    with _conn() as c:
        return c.execute(
            'SELECT contact_name, telegram_chat_id FROM contacts LIMIT 1'
        ).fetchone()


def load_bot_token():
    """Returns token string or None."""
    with _conn() as c:
        row = c.execute('SELECT bot_token FROM bot_config LIMIT 1').fetchone()
        return row[0] if row else None


def is_setup_complete():
    try:
        return (load_user() is not None
                and load_contact() is not None
                and load_bot_token() is not None)
    except Exception:
        return False


# ── Admin seed (pre-populated, skips CLI setup for the current user) ────────

def seed_admin(name, blood_group, medical_notes, contact_name,
               telegram_chat_id, bot_token):
    """Write all config in one call. Does nothing if already seeded."""
    init_db()
    if is_setup_complete():
        return
    save_user(name, blood_group, medical_notes)
    save_contact(contact_name, telegram_chat_id)
    save_bot_token(bot_token)
