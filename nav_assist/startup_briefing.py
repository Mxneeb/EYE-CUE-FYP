"""
Startup briefing — one-shot spoken summary at app launch.

Speaks, in a single natural sentence:
    greeting  +  time + date  +  current weather (if online)  +  battery status

Reuses the Piper / espeak pipeline and weather helpers from
`nav_assist.time_weather` so the voice and fallback behaviour match the
rest of the app. Blocks until speech finishes; intended to be launched
in a daemon thread from `app.py`.

Public API
----------
    from nav_assist.startup_briefing import announce_briefing
    announce_briefing()           # fetch, compose, speak; returns the sentence
"""

import datetime

from nav_assist.time_weather import (
    _fetch_weather, _compose_sentence,
    _speak_piper, _speak_espeak,
)

try:
    import psutil
except ImportError:
    psutil = None


def _greeting(hour):
    if 5 <= hour < 12:
        return 'Good morning.'
    if 12 <= hour < 17:
        return 'Good afternoon.'
    if 17 <= hour < 22:
        return 'Good evening.'
    return 'Hello.'


def _battery_phrase():
    """Return 'Battery 73 percent.' or '' if no battery info is available."""
    if psutil is None:
        return ''
    try:
        batt = psutil.sensors_battery()
    except Exception:
        return ''
    if batt is None or batt.percent is None:
        return ''
    pct = round(batt.percent)
    plugged = ' and charging' if getattr(batt, 'power_plugged', False) else ''
    return f'Battery {pct} percent{plugged}.'


def announce_briefing():
    """Compose and speak the startup briefing. Returns the spoken sentence."""
    now  = datetime.datetime.now()
    data = _fetch_weather()

    parts = [_greeting(now.hour), _compose_sentence(now, data)]
    batt = _battery_phrase()
    if batt:
        parts.append(batt)

    sentence = ' '.join(parts)
    print(f'[startup_briefing] {sentence}')

    if not _speak_piper(sentence):
        _speak_espeak(sentence)
    return sentence


if __name__ == '__main__':
    print(announce_briefing())
