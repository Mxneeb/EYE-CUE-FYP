"""
SOS trigger module.

Usage:
    from sos.sos import trigger_sos
    trigger_sos(camera_index=0)

Flow: beep → capture image → fetch location → load DB →
      send Telegram (text + photo + location pin) → speak confirmation.
Retries once on failure.
"""

import datetime
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import requests

# Allow running standalone (python3 sos/sos.py) or as a package import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db as _db

_GOOGLE_GEO_KEY   = 'AIzaSyB45XrgpSla6cLBbkJjQO38Nv5wvwEtQC4'
_GOOGLE_GEO_URL   = 'https://www.googleapis.com/geolocation/v1/geolocate'
_MLS_URL          = 'https://beacondb.net/v1/geolocate'   # fallback
_IPAPI_URL        = 'http://ip-api.com/json'               # last resort
_LOCATION_TIMEOUT = 5

# Telegram API fallback chain (tried in order):
#   1. Official endpoint
#   2. Your own Cloudflare Worker — paste your worker URL here after deploying
#      (dash.cloudflare.com → Workers & Pages → Create — see README or ask Claude)
#   3. Tor SOCKS5 on localhost — works if `tor` service is running
_YOUR_CF_WORKER = 'https://silent-tree-61c0.muneebpriv.workers.dev'

_TG_BASES = [b for b in [
    'https://api.telegram.org',
    _YOUR_CF_WORKER or None,
] if b]
_TOR_PROXY = {'https': 'socks5h://127.0.0.1:9050',
               'http':  'socks5h://127.0.0.1:9050'}


# ── Audio helpers ──────────────────────────────────────────────────────────

def _beep(times=3):
    """Play urgent alert tones via aplay (raw PCM, no external files needed)."""
    try:
        sr   = 22050
        dur  = 0.25
        freq = 880
        t     = np.linspace(0, dur, int(sr * dur), endpoint=False)
        tone  = (np.sin(2 * np.pi * freq * t) * 0.9 * 32767).astype(np.int16)
        gap   = np.zeros(int(sr * 0.1), dtype=np.int16)
        audio = np.tile(np.concatenate([tone, gap]), times).tobytes()
        proc  = subprocess.Popen(
            ['aplay', '-r', str(sr), '-f', 'S16_LE', '-c', '1', '-t', 'raw', '-'],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.communicate(audio)
    except Exception as exc:
        print(f'[SOS] Beep error: {exc}')


def _speak(text):
    """Speak via espeak — always available as last-resort TTS in an emergency."""
    safe = text.replace('"', "'")
    os.system(f'espeak-ng -s 150 "{safe}" 2>/dev/null || espeak -s 150 "{safe}" 2>/dev/null')


# ── Camera ─────────────────────────────────────────────────────────────────

def _capture_image(camera_index=0, frame=None):
    """
    Return JPEG bytes from the camera.
    If `frame` is provided (e.g. passed from the main nav loop), encode it
    directly — avoids opening a second VideoCapture on an already-held device.
    """
    if frame is not None:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    # Standalone mode: open camera ourselves
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('[SOS] Camera not available.')
        return None
    for _ in range(5):
        cap.grab()
    ret, grabbed = cap.read()
    cap.release()
    if not ret or grabbed is None:
        return None
    _, buf = cv2.imencode('.jpg', grabbed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


# ── Location ───────────────────────────────────────────────────────────────

def _scan_wifi_aps():
    """
    Use nmcli to list visible WiFi access points.
    Returns a list of {macAddress, signalStrength(dBm)} dicts for Beacondb.
    --rescan no uses cached results to avoid an 8-second RF scan block.
    """
    try:
        # Try fresh scan first; fall back to cached if it times out
        try:
            result = subprocess.run(
                ['nmcli', '--escape', 'no', '-t', '-f', 'BSSID,SIGNAL',
                 'dev', 'wifi', 'list', '--rescan', 'yes'],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            result = subprocess.run(
                ['nmcli', '--escape', 'no', '-t', '-f', 'BSSID,SIGNAL',
                 'dev', 'wifi', 'list', '--rescan', 'no'],
                capture_output=True, text=True, timeout=5,
            )
        aps = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                # rsplit from right so BSSID colons are preserved
                bssid, sig_str = line.rsplit(':', 1)
                sig_pct = int(sig_str)
                dbm = (sig_pct // 2) - 100   # nmcli 0-100% → approximate dBm
                aps.append({'macAddress': bssid, 'signalStrength': dbm})
            except (ValueError, IndexError):
                continue
        return aps
    except Exception as exc:
        print(f'[SOS] WiFi scan failed: {exc}')
        return []


def _get_location():
    """
    Location chain (best → worst accuracy):
      1. Google Geolocation API + WiFi scan  — 15–50 m
      2. Beacondb (Mozilla-compatible)       — fallback if Google fails
      3. ip-api.com IP geolocation           — ~1–10 km last resort
    """
    aps = _scan_wifi_aps()

    # ── 1. Google Geolocation API ─────────────────────────────────────────
    if aps:
        try:
            resp = requests.post(
                _GOOGLE_GEO_URL,
                params={'key': _GOOGLE_GEO_KEY},
                json={'wifiAccessPoints': aps},
                timeout=_LOCATION_TIMEOUT,
            )
            data = resp.json()
            loc  = data.get('location', {})
            lat  = loc.get('lat')
            lng  = loc.get('lng')
            acc  = data.get('accuracy')
            if lat and lng:
                print(f'[SOS] Google location: {lat:.6f}, {lng:.6f}  ±{acc:.0f} m')
                return lat, lng, f'GPS fix (±{acc:.0f} m)'
        except Exception as exc:
            print(f'[SOS] Google Geolocation failed: {exc}')

    # ── 2. Beacondb fallback ──────────────────────────────────────────────
    if aps:
        try:
            resp = requests.post(
                _MLS_URL,
                json={'wifiAccessPoints': aps},
                timeout=_LOCATION_TIMEOUT,
            )
            data = resp.json()
            loc  = data.get('location', {})
            lat  = loc.get('lat')
            lng  = loc.get('lng')
            acc  = data.get('accuracy')
            if lat and lng:
                print(f'[SOS] Beacondb location: {lat:.6f}, {lng:.6f}  ±{acc:.0f} m')
                return lat, lng, f'WiFi fix (±{acc:.0f} m)'
        except Exception as exc:
            print(f'[SOS] Beacondb failed: {exc}')

    # ── 3. IP geolocation last resort ─────────────────────────────────────
    try:
        data = requests.get(_IPAPI_URL, timeout=3).json()
        if data.get('status') == 'success':
            lat  = data.get('lat')
            lon  = data.get('lon')
            city = data.get('city', 'Unknown')
            print(f'[SOS] IP location: {city} ({lat}, {lon})')
            return lat, lon, city
    except Exception as exc:
        print(f'[SOS] IP geolocation failed: {exc}')

    return None, None, 'Location unavailable'


# ── Telegram sender ────────────────────────────────────────────────────────

def _tor_available():
    """Check if Tor's SOCKS5 port is listening on localhost."""
    import socket
    try:
        s = socket.create_connection(('127.0.0.1', 9050), timeout=1)
        s.close()
        return True
    except OSError:
        return False


def _tg_post(url, proxies=None, files=None, data=None, json_body=None):
    """Single POST to Telegram. Raises on HTTP error."""
    timeout = 60 if files else 30
    if files:
        r = requests.post(url, files=files, data=data or {},
                          proxies=proxies, timeout=timeout)
    else:
        r = requests.post(url, json=json_body or {},
                          proxies=proxies, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _tg(token, method, **kwargs):
    """
    POST to Telegram API with automatic fallback:
      1. Direct (api.telegram.org)
      2. Cloudflare public proxy (bypasses most ISP-level blocks, zero setup)
      3. Tor SOCKS5 via localhost:9050 (if tor service is running)
    Raises on total failure.
    """
    attempts = [(base, None) for base in _TG_BASES]
    if _tor_available():
        attempts.append((_TG_BASES[0], _TOR_PROXY))

    last_exc = None
    for base, proxies in attempts:
        url = f'{base}/bot{token}/{method}'
        label = ('Tor' if proxies else
                 'Cloudflare proxy' if 'pages.dev' in base else 'direct')
        try:
            result = _tg_post(url, proxies=proxies, **kwargs)
            print(f'[SOS] Telegram OK via {label}')
            return result
        except Exception as exc:
            print(f'[SOS] Telegram {label} failed: {exc}')
            last_exc = exc

    raise last_exc


def _send_alert(token, chat_id, user_info, image_bytes, lat, lon, city):
    """
    Send three Telegram messages in order:
      1. Formatted text alert
      2. Camera photo
      3. Location pin (if coordinates available)
    Returns True on success.
    """
    name, blood_group, medical_notes = user_info
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    loc_str = (f'{city} ({lat:.5f}, {lon:.5f})' if lat and lon
               else 'Location unavailable')

    text = (
        f'\U0001f198 *SOS ALERT* \u2014 {name} needs help!\n\n'
        f'\U0001f4cd Location: {loc_str}\n'
        f'\U0001f9b8 Blood group: {blood_group}\n'
        f'\U0001f3e5 Medical notes: {medical_notes}\n'
        f'\U0001f550 Time: {ts}'
    )

    _tg(token, 'sendMessage',
        json_body={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'})

    if image_bytes:
        _tg(token, 'sendPhoto',
            files={'photo': ('sos.jpg', image_bytes, 'image/jpeg')},
            data={'chat_id': chat_id, 'caption': '\U0001f4f7 Camera at time of SOS'})

    if lat and lon:
        _tg(token, 'sendLocation',
            json_body={'chat_id': chat_id, 'latitude': lat, 'longitude': lon})

    return True


# ── Public API ─────────────────────────────────────────────────────────────

def trigger_sos(camera_index=0, frame=None):
    """
    Main SOS trigger. Call this on emergency keypress.
    Pass `frame` (BGR ndarray) when calling from inside the nav app so the
    function doesn't try to open an already-held camera device.
    Non-blocking when called in a daemon thread.
    """
    print('[SOS] Triggered.')
    _beep()

    _db.init_db()
    user_info = _db.load_user()
    contact   = _db.load_contact()
    token     = _db.load_bot_token()

    if not user_info or not contact or not token:
        print('[SOS] Not configured. Run sos/setup.py first.')
        _speak('S O S failed. System not configured.')
        return

    contact_name, chat_id = contact

    image_bytes    = _capture_image(camera_index, frame=frame)
    lat, lon, city = _get_location()

    for attempt in range(2):
        try:
            _send_alert(token, chat_id, user_info, image_bytes, lat, lon, city)
            print(f'[SOS] Alert sent to {contact_name} ({chat_id}).')
            _speak(f'S O S sent to {contact_name}.')
            return
        except Exception as exc:
            print(f'[SOS] Attempt {attempt + 1} failed: {exc}')
            if attempt == 0:
                time.sleep(3)

    print('[SOS] All attempts failed.')
    _speak('S O S failed. Please call for help immediately.')


if __name__ == '__main__':
    trigger_sos()
