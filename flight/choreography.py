import numpy as np
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features

# Match these with main.py
Hz = 30  # visualizer runs at 30fps
TAKEOFF_HEIGHT = 1.0
BEAT_COOLDOWN = 0.6
BEAT_PULSE_HEIGHT = 0.4

def get_speed(t, rms_times, rms_norm):
    idx = np.searchsorted(rms_times, t)
    return rms_norm[min(idx, len(rms_norm) - 1)]

def get_size(t, freq_times, bass_norm):
    idx = np.searchsorted(freq_times, t)
    return bass_norm[min(idx, len(bass_norm) - 1)]

def get_drone1_positions(features, duration=40.0, fps=30):
    total_frames = int(duration * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= duration]

    # Normalize RMS to speed
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min())
    rms_norm = 0.5 + rms_norm * 2.5

    # Normalize bass to size
    bass_norm = (bass_energy - bass_energy.min()) / (bass_energy.max() - bass_energy.min())
    bass_norm = 0.5 + bass_norm * 1.0

    angles = []
    angle = 0.0
    for frame in range(total_frames):
        t = frame / fps
        speed = get_speed(t, rms_times, rms_norm)
        angle += speed * (2 * np.pi / fps) * 0.15
        angles.append(angle)

    angles = np.array(angles)
    x = np.array([get_size(f/fps, freq_times, bass_norm) * np.sin(angles[f]) for f in range(total_frames)])
    y = np.array([get_size(f/fps, freq_times, bass_norm) * np.sin(angles[f]) * np.cos(angles[f]) for f in range(total_frames)])
    z = np.full(total_frames, TAKEOFF_HEIGHT)

    # Beat pulse with cooldown and graceful decay
    last_beat_time = -BEAT_COOLDOWN
    for bt in beat_times:
        if (bt - last_beat_time) >= BEAT_COOLDOWN:
            frame_idx = int(bt * fps)
            for f in range(frame_idx, min(frame_idx + int(4 * fps / 20), total_frames)):
                time_since_beat = (f - frame_idx) / fps
                z[f] += BEAT_PULSE_HEIGHT * np.exp(-0.15 * (time_since_beat * 20))
            last_beat_time = bt

    # Clamp z
    z = np.clip(z, 0.5, 1.8)

    return x, y, z

def get_drone2_positions(features, duration=40.0, fps=30, start_time=20.0):
    total_frames = int(duration * fps)
    start_frame = int(start_time * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= start_time) & (beat_times <= duration)]

    rms_norm = (rms - rms.min()) / (rms.max() - rms.min())
    rms_norm = 0.5 + rms_norm * 2.5

    x = np.zeros(total_frames)
    y = np.zeros(total_frames)
    z = np.zeros(total_frames)

    orbit_radius = 1.0
    angle = 0.0
    for frame in range(start_frame, total_frames):
        t = frame / fps
        idx = np.searchsorted(rms_times, t)
        speed = rms_norm[min(idx, len(rms_norm) - 1)]
        angle -= speed * (2 * np.pi / fps) * 0.15
        x[frame] = orbit_radius * np.cos(angle)
        y[frame] = orbit_radius * np.sin(angle)
        z[frame] = 1.5

    # Beat pulse for drone 2
    last_beat_time = -BEAT_COOLDOWN
    for bt in beat_times:
        if (bt - last_beat_time) >= BEAT_COOLDOWN:
            frame_idx = int(bt * fps)
            for f in range(frame_idx, min(frame_idx + int(4 * fps / 20), total_frames)):
                time_since_beat = (f - frame_idx) / fps
                z[f] += BEAT_PULSE_HEIGHT * np.exp(-0.15 * (time_since_beat * 20))
            last_beat_time = bt

    z = np.clip(z, 0.5, 1.8)

    return x, y, z, start_frame