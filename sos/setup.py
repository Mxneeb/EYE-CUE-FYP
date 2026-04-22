#!/usr/bin/env python3
"""
SOS System First-Time Setup
Run once as caretaker: python3 sos/setup.py

Collects user info, emergency contact, and Telegram bot token.
Saves everything to ~/sos_system.db.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db


def _prompt(label, default=None):
    suffix = f' [{default}]' if default else ''
    val = input(f'{label}{suffix}: ').strip()
    return val if val else default


def main():
    print('\n╔══════════════════════════════╗')
    print('║    SOS System Setup          ║')
    print('╚══════════════════════════════╝\n')

    db.init_db()

    if db.is_setup_complete():
        print('⚠  A configuration already exists.')
        resp = input('Overwrite it? [y/N] ').strip().lower()
        if resp != 'y':
            print('Setup cancelled — existing config kept.')
            sys.exit(0)

    print('\n── User Information ──────────────────')
    name          = _prompt('Full name')
    blood_group   = _prompt('Blood group (e.g. O+, A-, B+)')
    medical_notes = _prompt('Medical notes (allergies, conditions — or press Enter to skip)', 'None')

    print('\n── Emergency Contact ─────────────────')
    contact_name = _prompt('Contact name')
    chat_id      = _prompt('Telegram chat ID of contact')

    print('\n── Telegram Bot ──────────────────────')
    bot_token = _prompt('Bot token (from @BotFather)')

    if not all([name, blood_group, contact_name, chat_id, bot_token]):
        print('\n✗ Required fields missing. Setup aborted.')
        sys.exit(1)

    db.save_user(name, blood_group, medical_notes)
    db.save_contact(contact_name, chat_id)
    db.save_bot_token(bot_token)

    print(f'\n✅ Setup complete! Database: {db.DB_PATH}')
    print('\n── How to get the Telegram chat ID ───')
    print('  1. Open Telegram and search for @userinfobot')
    print('  2. Send it any message — it replies with your user ID.')
    print('  3. That number is the chat ID for direct messages to the bot.')
    print('\n── How to create a Telegram bot ──────')
    print('  1. Open Telegram and search for @BotFather')
    print('  2. Send /newbot and follow the prompts.')
    print('  3. Copy the token it gives you (looks like 123456:ABC-DEF...).')
    print('\n  ⚠  The emergency contact must send /start to your bot first,')
    print('     otherwise Telegram will block messages from the bot to them.\n')


if __name__ == '__main__':
    main()
