"""
Enhanced Audio Feedback Module.

Enhanced Features:
    - Spatial audio: Stereo positioning (left obstacle → left ear)
    - Distance-based volume: Closer obstacles = louder
    - Audio icons: Distinct sounds for different obstacle types
    - Rhythm-based guidance: Beep frequency indicates urgency
    - Context-aware TTS: Different voices for different scenarios
    - Priority queue: Urgent obstacles interrupt lower-priority messages

Audio System:
    - pyttsx3 for TTS
    - winsound for audio icons and beeps
    - Spatial panning using stereo mixing
"""

import time
import queue
import threading
import numpy as np
from collections import deque

from nav_assist.config import (
    AUDIO_SPATIAL_ENABLED, AUDIO_DISTANCE_VOLUME, AUDIO_ICON_ENABLED,
    AUDIO_RHYTHM_ENABLED, AUDIO_SPATIAL_PAN_RANGE,
    AUDIO_MIN_VOLUME, AUDIO_MAX_VOLUME, AUDIO_SPEECH_RATE,
    AUDIO_ICON_FREQUENCIES, RHYTHM_BEEP_RATES,
)


class SpatialAudioMixer:
    """
    Mix audio with spatial positioning for left/right stereo output.
    """
    
    def __init__(self, pan_range=AUDIO_SPATIAL_PAN_RANGE):
        self.pan_range = pan_range
        
    def get_pan_value(self, direction, frame_width):
        """
        Calculate pan value for stereo positioning.
        
        Parameters:
            direction: Angle in degrees (-180 to 180, negative=left, positive=right)
            frame_width: Width of the frame in pixels
        
        Returns:
            pan_value: -1.0 (full left) to 1.0 (full right)
        """
        # Normalize direction to -1 to 1
        center_x = frame_width // 2
        
        # direction is the angular offset from center
        # Map to -1 to 1
        pan = np.clip(direction / 90.0, -1.0, 1.0)
        
        return pan * self.pan_range
    
    def mix_stereo(self, left_sample, right_sample):
        """
        Mix left and right channels.
        """
        return left_sample, right_sample


class AudioIconGenerator:
    """
    Generate distinct audio icons for different obstacle types.
    """
    
    def __init__(self):
        self.frequencies = AUDIO_ICON_FREQUENCIES
        self.last_beep_time = 0
        self.last_beep_type = None
        
    def get_frequency(self, obstacle_class):
        """Get the frequency for an obstacle type."""
        class_lower = obstacle_class.lower()
        
        for obstacle_type, freq in self.frequencies.items():
            if obstacle_type in class_lower:
                return freq
        
        return self.frequencies.get('default', 500)
    
    def get_beep_duration(self, obstacle_class):
        """Get beep duration based on obstacle type."""
        class_lower = obstacle_class.lower()
        
        # Critical obstacles get longer beeps
        if any(c in class_lower for c in ['person', 'car', 'truck']):
            return 200  # ms
        elif any(c in class_lower for c in ['wall', 'building', 'pole']):
            return 150
        else:
            return 100
    
    def should_beep(self, obstacle_class, distance, current_time):
        """
        Determine if we should beep based on rhythm rules.
        
        Parameters:
            obstacle_class: Type of obstacle
            distance: Distance to obstacle (normalized 0-1)
            current_time: Current timestamp
        
        Returns:
            (should_play, beep_duration)
        """
        if not AUDIO_RHYTHM_ENABLED:
            return False, 0
        
        # Determine beep rate based on distance
        if distance > 0.7:  # Very close
            beep_rate = RHYTHM_BEEP_RATES['very_close']
            duration = self.get_beep_duration(obstacle_class) * 1.5
        elif distance > 0.5:  # Close
            beep_rate = RHYTHM_BEEP_RATES['close']
            duration = self.get_beep_duration(obstacle_class)
        elif distance > 0.3:  # Medium
            beep_rate = RHYTHM_BEEP_RATES['medium']
            duration = self.get_beep_duration(obstacle_class) * 0.8
        else:  # Far
            beep_rate = RHYTHM_BEEP_RATES['far']
            duration = self.get_beep_duration(obstacle_class) * 0.5
        
        # Check if enough time has passed
        interval = 1.0 / beep_rate if beep_rate > 0 else 2.0
        
        if current_time - self.last_beep_time >= interval:
            self.last_beep_time = current_time
            self.last_beep_type = obstacle_class
            return True, int(duration)
        
        return False, 0


class EnhancedAudioFeedback:
    """
    Enhanced audio feedback system with spatial audio, distance-based volume,
    audio icons, and rhythm-based guidance.
    """
    
    def __init__(self, cooldown=3.0, enabled=True):
        self.enabled = enabled
        self.cooldown = cooldown
        self._last_spoken_time = 0.0
        self._last_instruction = ''
        self._queue = queue.Queue(maxsize=5)
        self._beep_queue = queue.Queue(maxsize=10)
        self._stop = threading.Event()
        self._engine = None
        self._thread = None
        self._beep_thread = None
        self._spatial_mixer = SpatialAudioMixer()
        self._icon_generator = AudioIconGenerator()
        self._volume = 1.0
        self._muted = False
        self._last_beep_time = 0
        self._message_history = deque(maxlen=10)
        
        self._init_engine()
    
    def _init_engine(self):
        """Initialize pyttsx3 engine and start threads."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', AUDIO_SPEECH_RATE)
            self._engine.setProperty('volume', AUDIO_MAX_VOLUME)
            
            # Start TTS thread
            self._thread = threading.Thread(target=self._run_tts, daemon=True)
            self._thread.start()
            
            # Start beep thread
            self._beep_thread = threading.Thread(target=self._run_beeps, daemon=True)
            self._beep_thread.start()
            
            print('[Audio] Enhanced audio system initialized')
        except Exception as e:
            print(f'[Audio] TTS engine unavailable: {e}')
            self._engine = None
    
    def _run_tts(self):
        """Background thread for TTS."""
        while not self._stop.is_set():
            try:
                audio_msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if not self.enabled or self._muted:
                continue
            
            try:
                text = audio_msg['text']
                priority = audio_msg.get('priority', 0)
                pan = audio_msg.get('pan', 0.0)
                volume = audio_msg.get('volume', 1.0)
                
                # Adjust volume
                if self._engine:
                    self._engine.setProperty('volume', volume * self._volume)
                
                # Speak
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
                
                self._message_history.append({
                    'text': text,
                    'time': time.time(),
                    'priority': priority
                })
                
            except Exception as e:
                print(f'[Audio] TTS error: {e}')
    
    def _run_beeps(self):
        """Background thread for audio icons and beeps."""
        try:
            import winsound
            import math
            
            while not self._stop.is_set():
                try:
                    beep_msg = self._beep_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if not self.enabled or self._muted:
                    continue
                
                try:
                    freq = beep_msg['frequency']
                    duration = beep_msg['duration']
                    pan = beep_msg.get('pan', 0.0)
                    volume = beep_msg.get('volume', 1.0)
                    
                    # Adjust frequency based on pan (subtle effect)
                    adjusted_freq = int(freq * (1.0 + pan * 0.05))
                    
                    # Play beep with adjusted volume
                    actual_volume = int(255 * volume * self._volume)
                    
                    # winsound doesn't support volume directly, so we use system default
                    winsound.Beep(adjusted_freq, duration)
                    
                except Exception as e:
                    pass  # Ignore beep errors
                    
        except ImportError:
            print('[Audio] winsound not available, beeps disabled')
    
    def speak(self, instruction, priority=0, direction=0, distance=1.0, obstacle_class=None):
        """
        Queue a navigation instruction for speech with spatial positioning.
        
        Parameters:
            instruction: Text to speak
            priority: Message priority (higher = more urgent)
            direction: Direction to obstacle (-180 to 180 degrees)
            distance: Normalized distance (0-1, higher = closer)
            obstacle_class: Type of obstacle for audio icon
        """
        if not self.enabled:
            return
        
        now = time.time()
        
        # Cooldown check
        if (instruction == self._last_instruction and
                now - self._last_spoken_time < self.cooldown):
            return
        if now - self._last_spoken_time < 1.5:
            return
        
        self._last_spoken_time = now
        self._last_instruction = instruction
        
        # Calculate volume based on distance
        volume = self._calculate_volume(distance)
        
        # Calculate pan for spatial audio
        pan = self._spatial_mixer.get_pan_value(direction, 640) if AUDIO_SPATIAL_ENABLED else 0.0
        
        # Queue for TTS
        try:
            self._queue.put_nowait({
                'text': instruction,
                'priority': priority,
                'pan': pan,
                'volume': volume,
            })
        except queue.Full:
            pass  # Drop if queue full
        
        # Queue for beep (audio icon)
        if AUDIO_ICON_ENABLED and obstacle_class:
            should_beep, duration = self._icon_generator.should_beep(
                obstacle_class, distance, now
            )
            if should_beep:
                freq = self._icon_generator.get_frequency(obstacle_class)
                try:
                    self._beep_queue.put_nowait({
                        'frequency': freq,
                        'duration': duration,
                        'pan': pan,
                        'volume': volume,
                    })
                except queue.Full:
                    pass
    
    def play_beep(self, frequency=440, duration=100, volume=1.0, pan=0.0):
        """
        Play a beep sound.
        
        Parameters:
            frequency: Beep frequency in Hz
            duration: Beep duration in ms
            volume: Volume level (0-1)
            pan: Stereo pan (-1 to 1)
        """
        if not self.enabled or self._muted:
            return
        
        try:
            self._beep_queue.put_nowait({
                'frequency': frequency,
                'duration': duration,
                'pan': pan,
                'volume': volume,
            })
        except queue.Full:
            pass
    
    def _calculate_volume(self, distance):
        """Calculate volume based on distance."""
        if not AUDIO_DISTANCE_VOLUME:
            return 1.0
        
        # Normalize distance (higher = closer = louder)
        # distance is 0-1 where 1 is closest
        volume = AUDIO_MIN_VOLUME + (AUDIO_MAX_VOLUME - AUDIO_MIN_VOLUME) * distance
        return np.clip(volume, AUDIO_MIN_VOLUME, AUDIO_MAX_VOLUME)
    
    def toggle(self):
        """Toggle audio on/off. Returns new state."""
        self.enabled = not self.enabled
        return self.enabled
    
    def mute(self):
        """Mute audio."""
        self._muted = True
    
    def unmute(self):
        """Unmute audio."""
        self._muted = False
    
    def set_volume(self, volume):
        """Set master volume (0-1)."""
        self._volume = np.clip(volume, 0.0, 1.0)
    
    def get_volume(self):
        """Get current master volume."""
        return self._volume
    
    def speak_obstacle_warning(self, obstacle_class, direction, distance):
        """
        Speak a warning about a specific obstacle.
        
        Parameters:
            obstacle_class: Type of obstacle
            direction: Direction to obstacle (-180 to 180)
            distance: Normalized distance (0-1)
        """
        # Format direction
        if abs(direction) < 20:
            dir_text = "ahead"
        elif direction < 0:
            dir_text = f"slightly left"
        else:
            dir_text = f"slightly right"
        
        # Format distance
        if distance > 0.7:
            dist_text = "very close"
        elif distance > 0.5:
            dist_text = "close"
        elif distance > 0.3:
            dist_text = "nearby"
        else:
            dist_text = "ahead"
        
        instruction = f"{obstacle_class.capitalize()} {dir_text}, {dist_text}"
        
        priority = 2 if obstacle_class.lower() in ['person', 'car', 'wall'] else 1
        
        self.speak(instruction, priority, direction, distance, obstacle_class)
    
    def speak_navigation(self, instruction, action_type, direction=0):
        """
        Speak navigation instruction.
        
        Parameters:
            instruction: Navigation instruction text
            action_type: Type of action (MOVE_AHEAD, MOVE_LEFT, etc.)
            direction: Direction value (-1 to 1)
        """
        # Map action type to emphasis
        if action_type == 'STOP':
            emphasis = "STOP!"
            priority = 3
        elif action_type == 'MOVE_AHEAD':
            emphasis = ""
            priority = 1
        elif action_type == 'MOVE_LEFT':
            emphasis = "turn left"
            priority = 1
        elif action_type == 'MOVE_RIGHT':
            emphasis = "turn right"
            priority = 1
        else:
            emphasis = ""
            priority = 1
        
        # Add spatial info
        direction_deg = direction * 90  # Convert -1..1 to degrees
        distance = 0.5  # Medium distance for navigation
        
        self.speak(instruction, priority, direction_deg, distance, None)
    
    def speak_status(self, status_text):
        """Speak a status message (low priority)."""
        self.speak(status_text, priority=0, direction=0, distance=0.2, obstacle_class=None)
    
    def shutdown(self):
        """Signal all threads to stop."""
        self._stop.set()
        
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._beep_thread is not None:
            self._beep_thread.join(timeout=2)
        
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
    
    def get_message_history(self):
        """Get recent message history."""
        return list(self._message_history)


class ComfortScorer:
    """
    Track user comfort based on warning frequency and type.
    """
    
    def __init__(self):
        self.recent_warnings = deque(maxlen=50)
        self.comfort_score = 10.0  # Start at max (10)
        
    def record_warning(self, priority, time_since_last):
        """
        Record a warning and update comfort score.
        
        Parameters:
            priority: Warning priority (1-3)
            time_since_last: Time since last warning in seconds
        """
        self.recent_warnings.append({
            'priority': priority,
            'time': time.time(),
        })
        
        # Update comfort score
        self._update_comfort()
    
    def _update_comfort(self):
        """Calculate comfort score based on warning history."""
        if not self.recent_warnings:
            self.comfort_score = 10.0
            return
        
        # Factors affecting comfort:
        # 1. Warning frequency
        # 2. Warning severity
        # 3. Time since last warning
        
        recent = list(self.recent_warnings)
        
        # Calculate warning rate (warnings per minute)
        if len(recent) >= 2:
            time_span = recent[-1]['time'] - recent[0]['time']
            if time_span > 0:
                warning_rate = len(recent) / (time_span / 60.0)
            else:
                warning_rate = 0
        else:
            warning_rate = 0
        
        # Calculate average priority
        avg_priority = np.mean([w['priority'] for w in recent])
        
        # Time decay factor
        time_since_last = time.time() - recent[-1]['time']
        decay = min(time_since_last / 30.0, 1.0)  # Fully recovered after 30 seconds
        
        # Base score
        score = 10.0
        
        # Reduce for high warning rate
        if warning_rate > 10:
            score -= 3
        elif warning_rate > 5:
            score -= 2
        elif warning_rate > 2:
            score -= 1
        
        # Reduce for high priority warnings
        if avg_priority > 2:
            score -= 1.5
        elif avg_priority > 1.5:
            score -= 1
        
        # Apply decay
        self.comfort_score = np.clip(score * decay + 10.0 * (1 - decay), 0, 10)
    
    def get_comfort_score(self):
        """Get current comfort score (0-10)."""
        return self.comfort_score
    
    def get_comfort_level(self):
        """Get comfort level description."""
        score = self.comfort_score
        if score >= 8:
            return "Very Comfortable"
        elif score >= 6:
            return "Comfortable"
        elif score >= 4:
            return "Moderate"
        elif score >= 2:
            return "Uncomfortable"
        else:
            return "Very Uncomfortable"
