import numpy as np
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features

def get_drone1_positions(features, duration=40.0, fps=30):
    """
    Drone 1 - Figure-8 pattern, size driven by bass energy,
    speed driven by RMS loudness.
    """
    total_frames = int(duration * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    freq_times = features['freq_times']

    # Normalize RMS to speed multiplier
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min())
    rms_norm = 0.5 + rms_norm * 2.5

    # Normalize bass to size multiplier (0.5 to 1.5)
    bass_norm = (bass_energy - bass_energy.min()) / (bass_energy.max() - bass_energy.min())
    bass_norm = 0.5 + bass_norm * 1.0

    def get_speed(t):
        idx = np.searchsorted(rms_times, t)
        return rms_norm[min(idx, len(rms_norm) - 1)]

    def get_size(t):
        idx = np.searchsorted(freq_times, t)
        return bass_norm[min(idx, len(bass_norm) - 1)]

    # Figure-8 uses lemniscate formula
    angles = []
    angle = 0.0
    for frame in range(total_frames):
        t = frame / fps
        speed = get_speed(t)
        angle += speed * (2 * np.pi / fps) * 0.15
        angles.append(angle)

    angles = np.array(angles)
    x = np.array([get_size(f/fps) * np.sin(angles[f]) for f in range(total_frames)])
    y = np.array([get_size(f/fps) * np.sin(angles[f]) * np.cos(angles[f]) for f in range(total_frames)])
    z = np.full(total_frames, 1.0)

    # Pulse height on beats
    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= duration]
    for bt in beat_times:
        frame_idx = int(bt * fps)
        for f in range(frame_idx, min(frame_idx + 8, total_frames)):
            z[f] += 0.3 * np.exp(-0.5 * (f - frame_idx))

    return x, y, z


def get_drone2_positions(features, duration=40.0, fps=30, start_time=20.0):
    """
    Drone 2 - Joins at start_time, orbits opposite direction to drone 1
    at a different height, speed also driven by RMS.
    """
    total_frames = int(duration * fps)
    start_frame = int(start_time * fps)
    rms_times = features['rms_times']
    rms = features['rms']

    # Normalize RMS
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min())
    rms_norm = 0.5 + rms_norm * 2.5

    def get_speed(t):
        idx = np.searchsorted(rms_times, t)
        return rms_norm[min(idx, len(rms_norm) - 1)]

    x = np.zeros(total_frames)
    y = np.zeros(total_frames)
    z = np.zeros(total_frames)

    # Only compute positions after start_time
    orbit_radius = 1.0
    angle = 0.0
    for frame in range(start_frame, total_frames):
        t = frame / fps
        speed = get_speed(t)
        # Negative angle = opposite direction to drone 1
        angle -= speed * (2 * np.pi / fps) * 0.15
        x[frame] = orbit_radius * np.cos(angle)
        y[frame] = orbit_radius * np.sin(angle)
        z[frame] = 1.5  # higher than drone 1

    # Pulse on beats
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= start_time) & (beat_times <= duration)]
    for bt in beat_times:
        frame_idx = int(bt * fps)
        for f in range(frame_idx, min(frame_idx + 8, total_frames)):
            z[f] += 0.2 * np.exp(-0.5 * (f - frame_idx))

    return x, y, z, start_frame