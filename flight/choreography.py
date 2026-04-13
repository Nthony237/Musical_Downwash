import numpy as np
import sys
from scipy.ndimage import uniform_filter1d
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
TAKEOFF_HEIGHT = 1.0
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.15
BEAT_PULSE_DURATION = 0.8
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 4.0
SPIRAL_HEIGHT = 0.5
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 0.8
ENABLE_SPIRAL = True
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
Z_MIN, Z_MAX = 0.5, 2.2

def clamp_position(x, y, z):
    x = np.clip(x, X_MIN, X_MAX)
    y = np.clip(y, Y_MIN, Y_MAX)
    z = np.clip(z, Z_MIN, Z_MAX)
    return x, y, z

def smooth(signal, window=60):
    """Smooth a signal using a rolling average to remove micro-fluctuations."""
    return uniform_filter1d(signal.astype(float), size=window)

def get_value(t, times, values):
    idx = np.searchsorted(times, t)
    return values[min(idx, len(values) - 1)]

def get_beat_pulse(t_since_beat, duration=0.8, height=0.15):
    """Smooth sine arch — rises and falls gracefully over duration seconds."""
    if t_since_beat < 0 or t_since_beat > duration:
        return 0.0
    progress = t_since_beat / duration
    return height * np.sin(progress * np.pi)

def get_spiral_offset(t_since_drop):
    """
    Spiral starts tiny, grows outward over first 80%,
    then fades smoothly back to zero so there's no snap when it ends.
    """
    if t_since_drop < 0 or t_since_drop > SPIRAL_DURATION:
        return 0.0, 0.0, 0.0

    progress = t_since_drop / SPIRAL_DURATION
    spiral_angle = progress * 6 * np.pi  # 3 full rotations

    # Height arch
    z_offset = SPIRAL_HEIGHT * np.sin(progress * np.pi)

    # Radius grows then shrinks back to zero in last 20%
    if progress < 0.8:
        radius = SPIRAL_RADIUS_START + (SPIRAL_RADIUS_END - SPIRAL_RADIUS_START) * (progress / 0.8)
    else:
        fade = (progress - 0.8) / 0.2   # 0 to 1 in last 20%
        radius = SPIRAL_RADIUS_END * (1 - fade)

    x_offset = radius * np.cos(spiral_angle)
    y_offset = radius * np.sin(spiral_angle)

    return x_offset, y_offset, z_offset

def get_drone1_positions(features, duration=40.0, fps=30):
    total_frames = int(duration * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= duration]
    drop_times = features['drop_times']
    drop_times = drop_times[drop_times <= duration]

    # Smooth heavily to remove micro-fluctuations
    rms_smooth = smooth(rms, window=60)
    bass_smooth = smooth(bass_energy, window=60)

    # Normalize
    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6        # range: 0.7 to 1.3

    bass_norm = (bass_smooth - bass_smooth.min()) / (bass_smooth.max() - bass_smooth.min())
    bass_norm = 0.5 + bass_norm * 1.0      # range: 0.5 to 1.5

    x_pos = np.zeros(total_frames)
    y_pos = np.zeros(total_frames)
    z_pos = np.full(total_frames, TAKEOFF_HEIGHT)

    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN

    # Landing approach starts 3 seconds before end
    land_start = duration - 3.0

    for frame in range(total_frames):
        t = frame / fps
        speed = get_value(t, rms_times, rms_norm)
        size = get_value(t, freq_times, bass_norm)

        # Slightly slower graceful figure-8
        angle += speed * (2 * np.pi / fps) * 0.032

        x = size * np.sin(angle)
        y = size * np.sin(angle) * np.cos(angle)
        z = TAKEOFF_HEIGHT

        # Beat pulse — smooth sine arch
        for bt in beat_times:
            time_since_beat = t - bt
            if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                    z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                    if time_since_beat < (1 / fps):
                        last_beat_time = bt
                break

        # Spiral on beat drops
        if ENABLE_SPIRAL:
            for dt in drop_times:
                if abs(t - dt) < (1 / fps):
                    if (dt - last_spiral_time) >= SPIRAL_COOLDOWN:
                        current_spiral_start = t
                        last_spiral_time = t
                    break

            t_since_spiral = t - current_spiral_start
            if 0 <= t_since_spiral <= SPIRAL_DURATION:
                sx, sy, sz = get_spiral_offset(t_since_spiral)
                x += sx
                y += sy
                z += sz

        # Landing approach — drift to center and descend over last 3 seconds
        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = TAKEOFF_HEIGHT * (1 - land_progress) + Z_MIN * land_progress

        x, y, z = clamp_position(x, y, z)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z

    return x_pos, y_pos, z_pos

def get_drone2_positions(features, duration=40.0, fps=30, start_time=20.0):
    total_frames = int(duration * fps)
    start_frame = int(start_time * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= start_time) & (beat_times <= duration)]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= start_time) & (drop_times <= duration)]

    rms_smooth = smooth(rms, window=60)
    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6

    x_pos = np.zeros(total_frames)
    y_pos = np.zeros(total_frames)
    z_pos = np.zeros(total_frames)

    orbit_radius = 1.0
    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN

    land_start = duration - 3.0

    for frame in range(start_frame, total_frames):
        t = frame / fps
        idx = np.searchsorted(rms_times, t)
        speed = rms_norm[min(idx, len(rms_norm) - 1)]

        angle -= speed * (2 * np.pi / fps) * 0.032
        x = orbit_radius * np.cos(angle)
        y = orbit_radius * np.sin(angle)
        z = 1.5

        # Beat pulse
        for bt in beat_times:
            time_since_beat = t - bt
            if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                    z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                    if time_since_beat < (1 / fps):
                        last_beat_time = bt
                break

        # Spiral
        if ENABLE_SPIRAL:
            for dt in drop_times:
                if abs(t - dt) < (1 / fps):
                    if (dt - last_spiral_time) >= SPIRAL_COOLDOWN:
                        current_spiral_start = t
                        last_spiral_time = t
                    break

            t_since_spiral = t - current_spiral_start
            if 0 <= t_since_spiral <= SPIRAL_DURATION:
                sx, sy, sz = get_spiral_offset(t_since_spiral)
                x += sx
                y += sy
                z += sz

        # Landing
        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = 1.5 * (1 - land_progress) + Z_MIN * land_progress

        x, y, z = clamp_position(x, y, z)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z

    return x_pos, y_pos, z_pos, start_frame