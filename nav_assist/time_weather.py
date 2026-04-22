"""
Time and weather announcement module for a blind user.

Fetches the local time offline (Python datetime) and today's forecast from
Open-Meteo (no API key). Builds a short, human-sounding report and speaks it
via Piper TTS, falling back to espeak if Piper is not installed. Audio is
played through aplay.

Public API
----------
    from nav_assist.time_weather import get_weather_and_time
    get_weather_and_time()   # fetch, compose, speak; returns the sentence

Network behaviour
-----------------
Any Open-Meteo failure (5 s timeout, HTTP error, JSON error) downgrades
gracefully to a time-only announcement.
"""

import datetime
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request

# ── Location (edit these for your deployment) ──────────────────────────────
LATITUDE  = 33.6844      # Islamabad, Pakistan
LONGITUDE = 73.0479

# ── Open-Meteo endpoint + hard timeout ─────────────────────────────────────
_API_URL          = 'https://api.open-meteo.com/v1/forecast'
_API_TIMEOUT_SECS = 5.0

# ── Time-of-day buckets: (start_hr inclusive, end_hr exclusive, label) ─────
_PERIODS = [
    (6,  12, 'morning'),
    (12, 17, 'afternoon'),
    (17, 21, 'evening'),
    (21, 24, 'night'),
]

# ── WMO weather codes → adjective-style conditions ─────────────────────────
_WEATHER_CODES = {
    0:  'clear',
    1:  'mostly clear',
    2:  'partly cloudy',
    3:  'overcast',
    45: 'foggy',
    48: 'foggy',
    51: 'drizzling',
    53: 'drizzling',
    55: 'heavy drizzle',
    56: 'freezing drizzle',
    57: 'freezing drizzle',
    61: 'light rain',
    63: 'raining',
    65: 'heavy rain',
    66: 'freezing rain',
    67: 'freezing rain',
    71: 'light snow',
    73: 'snowing',
    75: 'heavy snow',
    77: 'snow grains',
    80: 'light showers',
    81: 'showers',
    82: 'heavy showers',
    85: 'light snow showers',
    86: 'snow showers',
    95: 'thunderstorms',
    96: 'thunderstorms and hail',
    99: 'severe thunderstorms and hail',
}

# Conditions that flow as "a {temp} and {cond} {label}"; others use "with {cond}".
_ADJECTIVE_CONDITIONS = {
    'clear', 'mostly clear', 'partly cloudy', 'overcast',
    'foggy', 'drizzling', 'raining', 'snowing', 'unsettled',
}

# Codes whose human label already implies precipitation — skip the rain phrase.
_PRECIP_CODES = set(range(50, 100))

# ── Piper voice: user-specified primary, project voice as secondary ────────
_PIPER_PRIMARY_MODEL = 'en_US-lessac-medium'
try:
    from nav_assist.config import PIPER_VOICE_ONNX as _PROJECT_VOICE
except Exception:
    _PROJECT_VOICE = None


# ── Time formatting ────────────────────────────────────────────────────────

# Hardcoded English names — strftime %A/%B/%p are locale-sensitive and will
# produce Arabic/Urdu strings if the system locale is not en_US.
_EN_DAYS   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
               'Friday', 'Saturday', 'Sunday']
_EN_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']


def _format_time(now):
    """'9:15 AM on Friday, April 17' — locale-independent."""
    hour12 = now.hour % 12 or 12
    ampm   = 'AM' if now.hour < 12 else 'PM'
    day    = _EN_DAYS[now.weekday()]
    month  = _EN_MONTHS[now.month - 1]
    return f'{hour12}:{now.minute:02d} {ampm} on {day}, {month} {now.day}'


# ── Weather fetch ──────────────────────────────────────────────────────────

def _fetch_weather():
    params = {
        'latitude':        LATITUDE,
        'longitude':       LONGITUDE,
        'current_weather': 'true',
        'hourly':          'temperature_2m,precipitation_probability,weathercode',
        'timezone':        'auto',
        'forecast_days':   1,
    }
    url = f'{_API_URL}?{urllib.parse.urlencode(params)}'
    try:
        with urllib.request.urlopen(url, timeout=_API_TIMEOUT_SECS) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as exc:
        print(f'[time_weather] Open-Meteo fetch failed: {exc}')
        return None


# ── Period aggregation ─────────────────────────────────────────────────────

def _remaining_periods(data, now):
    hourly = data.get('hourly') or {}
    times  = hourly.get('time') or []
    temps  = hourly.get('temperature_2m') or []
    precs  = hourly.get('precipitation_probability') or []
    codes  = hourly.get('weathercode') or []
    if not (times and temps and codes):
        return []

    current_hour = now.hour
    out = []
    for start, end, label in _PERIODS:
        # Skip past periods and the one we're currently in — "right now" covers it.
        if end <= current_hour or start <= current_hour < end:
            continue

        idxs = []
        for i, iso in enumerate(times):
            try:
                hour = int(iso[11:13])
            except (ValueError, IndexError):
                continue
            if start <= hour < end:
                idxs.append(i)
        if not idxs:
            continue

        avg_temp  = sum(temps[i] for i in idxs) / len(idxs)
        peak_prec = max((precs[i] for i in idxs if precs[i] is not None), default=0)
        counts = {}
        for i in idxs:
            counts[codes[i]] = counts.get(codes[i], 0) + 1
        # Most frequent code; ties broken by larger value (higher severity).
        dominant = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        out.append({
            'label':     label,
            'avg_temp':  avg_temp,
            'peak_prec': peak_prec,
            'code':      dominant,
        })
    return out


# ── Descriptors ────────────────────────────────────────────────────────────

def _temp_word(c):
    if c < 0:  return 'freezing'
    if c < 10: return 'cold'
    if c < 18: return 'cool'
    if c < 24: return 'mild'
    if c < 30: return 'warm'
    return 'hot'


def _rain_phrase(pct):
    if pct is None or pct < 20:
        return ''
    if pct < 50:
        return 'a slight chance of rain'
    if pct < 80:
        return 'a good chance of rain'
    return 'rain likely'


def _condition(code):
    return _WEATHER_CODES.get(code, 'unsettled')


# ── Period merging ─────────────────────────────────────────────────────────

def _merge(periods):
    """Collapse adjacent periods that share temp bucket, condition, and rain class."""
    if not periods:
        return []
    out = [dict(periods[0])]
    for p in periods[1:]:
        last = out[-1]
        if (_temp_word(last['avg_temp'])         == _temp_word(p['avg_temp'])
                and _condition(last['code'])      == _condition(p['code'])
                and _rain_phrase(last['peak_prec']) == _rain_phrase(p['peak_prec'])):
            last['label']     = f"{last['label']} and {p['label']}"
            last['avg_temp']  = (last['avg_temp'] + p['avg_temp']) / 2
            last['peak_prec'] = max(last['peak_prec'], p['peak_prec'])
        else:
            out.append(dict(p))
    return out


# ── Period description ─────────────────────────────────────────────────────

def _describe_period(p):
    temp_w = _temp_word(p['avg_temp'])
    cond   = _condition(p['code'])
    if cond in _ADJECTIVE_CONDITIONS:
        phrase = f'a {temp_w} and {cond} {p["label"]}'
    else:
        phrase = f'a {temp_w} {p["label"]} with {cond}'
    # Skip rain probability if the condition itself already implies precipitation.
    if p['code'] not in _PRECIP_CODES:
        rain = _rain_phrase(p['peak_prec'])
        if rain:
            phrase += f' with {rain}'
    return phrase


def _join(items):
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


# ── Sentence composition ───────────────────────────────────────────────────

def _compose_sentence(now, data):
    time_str = _format_time(now)
    if not data:
        return f'It is {time_str}. Weather information is currently unavailable.'

    parts = [f'It is {time_str}.']

    current = data.get('current_weather') or {}
    temp = current.get('temperature')
    code = current.get('weathercode')
    if temp is not None:
        cond = _condition(code) if code is not None else 'clear'
        parts.append(
            f"Right now it's {_temp_word(temp)} and {cond} at {round(temp)} degrees."
        )

    periods = _merge(_remaining_periods(data, now))
    if periods:
        parts.append(f'Expect {_join([_describe_period(p) for p in periods])}.')

    return ' '.join(parts)


# ── TTS backends ───────────────────────────────────────────────────────────

def _speak(text):
    print(f'[time_weather] Speaking: {text}')
    if _speak_piper(text):
        return
    if _speak_espeak(text):
        return
    print('[time_weather] No working TTS backend; report not spoken.')


def _speak_piper(text):
    """Return True if the sentence was synthesized and played via Piper."""
    if shutil.which('piper') is None:
        return False
    candidates = [_PIPER_PRIMARY_MODEL]
    if _PROJECT_VOICE and os.path.exists(_PROJECT_VOICE):
        candidates.append(_PROJECT_VOICE)
    for model in candidates:
        if _run_piper_pipe(text, model):
            return True
    return False


def _run_piper_pipe(text, model):
    """piper --model <model> --output_raw  |  aplay -r 22050 -f S16_LE -t raw -"""
    piper = aplay = None
    try:
        piper = subprocess.Popen(
            ['piper', '--model', model, '--output_raw'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        aplay = subprocess.Popen(
            ['aplay', '-q', '-r', '22050', '-f', 'S16_LE', '-t', 'raw', '-'],
            stdin=piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        piper.stdout.close()            # let piper receive SIGPIPE if aplay exits
        piper.stdin.write(text.encode('utf-8'))
        piper.stdin.close()
        aplay.wait(timeout=30)
        piper.wait(timeout=5)
        return piper.returncode == 0 and aplay.returncode == 0
    except Exception as exc:
        print(f'[time_weather] Piper pipeline error ({model}): {exc}')
        for p in (piper, aplay):
            if p is not None:
                try: p.kill()
                except Exception: pass
        return False


def _speak_espeak(text):
    binary = shutil.which('espeak-ng') or shutil.which('espeak')
    if binary is None:
        return False
    try:
        subprocess.run(
            [binary, '-s', '160', text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception as exc:
        print(f'[time_weather] espeak error: {exc}')
        return False


# ── Public entry points ────────────────────────────────────────────────────

def announce_time():
    """Read system clock, speak the time and date. No network call. Returns the sentence."""
    now = datetime.datetime.now()
    sentence = f'It is {_format_time(now)}.'
    print(f'[time_weather] Speaking: {sentence}')
    if not _speak_piper(sentence):
        _speak_espeak(sentence)
    return sentence


def get_weather_and_time():
    """Fetch time + weather, compose a natural report, speak it. Returns the sentence."""
    now = datetime.datetime.now()
    data = _fetch_weather()
    sentence = _compose_sentence(now, data)
    _speak(sentence)
    return sentence


if __name__ == '__main__':
    print(get_weather_and_time())
