import numpy as np
import sys
from scipy.ndimage import uniform_filter1d
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
TAKEOFF_HEIGHT = 1.0
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.30
BEAT_PULSE_DURATION = 0.8
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 4.0
SPIRAL_HEIGHT = 1.0
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 1.8
ENABLE_SPIRAL = True
BLEND_DURATION = 5.0

# ---- HEIGHT ZONES ----
D1_S1_Z_MIN, D1_S1_Z_MAX = 0.5, 2.5
D1_S2_Z_MIN, D1_S2_Z_MAX = 1.5, 2.5
D2_S2_Z_MIN, D2_S2_Z_MAX = 0.5, 1.4

X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0
Z_MIN, Z_MAX = 0.5, 2.5

# ---- INTERACTION SETTINGS ----
INTERACT_DURATION = 8.0    # total seconds for spiral toward + apart
INTERACT_RADIUS = 1.0      # how far apart they start the spiral
MEET_HEIGHT = 1.5          # height where they meet in the middle

def clamp(x, y, z, z_min=Z_MIN, z_max=Z_MAX):
    x = np.clip(x, X_MIN, X_MAX)
    y = np.clip(y, Y_MIN, Y_MAX)
    z = np.clip(z, z_min, z_max)
    return x, y, z

def smooth(signal, window=60):
    return uniform_filter1d(signal.astype(float), size=window)

def get_value(t, times, values):
    idx = np.searchsorted(times, t)
    return values[min(idx, len(values) - 1)]

def get_beat_pulse(t_since_beat, duration=0.8, height=0.30):
    if t_since_beat < 0 or t_since_beat > duration:
        return 0.0
    progress = t_since_beat / duration
    return height * np.sin(progress * np.pi)

def get_spiral_offset(t_since_drop):
    if t_since_drop < 0 or t_since_drop > SPIRAL_DURATION:
        return 0.0, 0.0, 0.0
    progress = t_since_drop / SPIRAL_DURATION
    spiral_angle = progress * 8 * np.pi
    z_offset = SPIRAL_HEIGHT * progress
    radius = SPIRAL_RADIUS_START + (SPIRAL_RADIUS_END - SPIRAL_RADIUS_START) * progress
    x_offset = radius * np.cos(spiral_angle)
    y_offset = radius * np.sin(spiral_angle)
    return x_offset, y_offset, z_offset

def figure8(angle, size):
    x = size * np.sin(angle)
    y = size * np.sin(angle) * np.cos(angle)
    return x, y

def lissajous(angle, size):
    x = size * np.sin(angle)
    y = size * np.sin(2 * angle + np.pi / 4)
    return x, y

def blend_shapes(pos_a, pos_b, factor):
    return pos_a * (1 - factor) + pos_b * factor

def get_interaction_offset(t_since_interact, drone_id):
    """
    Both drones spiral toward center point, meet, then spiral apart.
    First half: spiral inward. Second half: spiral outward.
    Drone 1 comes from above, drone 2 comes from below.
    They meet at MEET_HEIGHT.
    """
    if t_since_interact < 0 or t_since_interact > INTERACT_DURATION:
        return 0.0, 0.0, 0.0

    progress = t_since_interact / INTERACT_DURATION
    half = INTERACT_DURATION / 2.0
    t_half = t_since_interact / half  # 0 to 1 for each half

    if t_since_interact <= half:
        # Spiral inward — radius shrinks to 0
        inward_progress = t_since_interact / half
        radius = INTERACT_RADIUS * (1 - inward_progress)
        spiral_angle = inward_progress * 4 * np.pi  # 2 rotations inward

        if drone_id == 1:
            # Drone 1 descends from upper zone to meet height
            z_offset = -(1.8 - MEET_HEIGHT) * inward_progress
        else:
            # Drone 2 rises from lower zone to meet height
            z_offset = (MEET_HEIGHT - 0.9) * inward_progress

    else:
        # Spiral outward — radius grows back
        outward_progress = (t_since_interact - half) / half
        radius = INTERACT_RADIUS * outward_progress
        spiral_angle = outward_progress * 4 * np.pi + 4 * np.pi  # continue rotation

        if drone_id == 1:
            # Drone 1 ascends back to upper zone
            z_offset = -(1.8 - MEET_HEIGHT) * (1 - outward_progress)
        else:
            # Drone 2 descends back to lower zone
            z_offset = (MEET_HEIGHT - 0.9) * (1 - outward_progress)

    # Opposite sides of the orbit — drone 2 is pi radians offset
    if drone_id == 2:
        spiral_angle += np.pi

    x_offset = radius * np.cos(spiral_angle)
    y_offset = radius * np.sin(spiral_angle)

    return x_offset, y_offset, z_offset

def find_biggest_drop(drop_times, onset_strength, rms_times, section_start, section_end):
    """Find the single biggest onset drop in section 2."""
    mask = (drop_times >= section_start) & (drop_times <= section_end)
    section_drops = drop_times[mask]

    if len(section_drops) == 0:
        # fallback to midpoint of section
        return (section_start + section_end) / 2

    # Find which drop has highest onset strength
    best_time = section_drops[0]
    best_strength = 0

    for dt in section_drops:
        idx = np.searchsorted(rms_times, dt)
        idx = min(idx, len(onset_strength) - 1)
        strength = onset_strength[idx]
        if strength > best_strength:
            best_strength = strength
            best_time = dt

    return best_time

def get_drone1_positions(features, duration=111.0, fps=30):
    total_frames = int(duration * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    treble_energy = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= duration]
    drop_times = features['drop_times']
    drop_times = drop_times[drop_times <= duration]
    section_times = features['section_times']
    s1_end = section_times[1]
    s2_end = section_times[2]

    rms_smooth = smooth(rms, window=60)
    bass_smooth = smooth(bass_energy, window=60)
    treble_smooth = smooth(treble_energy, window=60)

    # Raw onset strength for finding biggest drop
    import librosa
    y, sr = librosa.load(features.get('filepath', ''), sr=None) if 'filepath' in features else (None, None)

    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6

    bass_norm = (bass_smooth - bass_smooth.min()) / (bass_smooth.max() - bass_smooth.min())
    bass_norm = 0.5 + bass_norm * 1.0

    treble_norm = (treble_smooth - treble_smooth.min()) / (treble_smooth.max() - treble_smooth.min())
    treble_norm = 0.0 + treble_norm * 0.3

    # Find biggest drop in section 2 for interaction trigger
    onset_strength = features.get('onset_strength', rms_smooth)
    interact_time = find_biggest_drop(drop_times, onset_strength, rms_times, s1_end, s2_end)
    print(f"Interaction triggered at t={interact_time:.1f}s")

    x_pos = np.zeros(total_frames)
    y_pos = np.zeros(total_frames)
    z_pos = np.full(total_frames, TAKEOFF_HEIGHT)

    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN
    current_interact_start = -INTERACT_DURATION
    land_start = duration - 3.0

    for frame in range(total_frames):
        t = frame / fps
        speed = get_value(t, rms_times, rms_norm)
        size = get_value(t, freq_times, bass_norm)
        treble_height = get_value(t, freq_times, treble_norm)

        in_interaction = (t >= interact_time and
                         t <= interact_time + INTERACT_DURATION and
                         t >= s1_end)

        # ---- SECTION 1 ----
        if t <= s1_end:
            angle += speed * (2 * np.pi / fps) * 0.032
            x, y = figure8(angle, size)
            z = TAKEOFF_HEIGHT + treble_height
            z_min, z_max = D1_S1_Z_MIN, D1_S1_Z_MAX

            for bt in beat_times:
                time_since_beat = t - bt
                if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                    if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                        z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                        if time_since_beat < (1 / fps):
                            last_beat_time = bt
                    break

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

        # ---- SECTION 2 ----
        else:
            angle += speed * (2 * np.pi / fps) * 0.032

            blend_factor = min((t - s1_end) / BLEND_DURATION, 1.0)
            f8x, f8y = figure8(angle, size)
            lsx, lsy = lissajous(angle, size)
            x = blend_shapes(f8x, lsx, blend_factor)
            y = blend_shapes(f8y, lsy, blend_factor)
            z = 1.8 + treble_height
            z_min, z_max = D1_S2_Z_MIN, D1_S2_Z_MAX

            if not in_interaction:
                # Normal beat pulse and spiral
                for bt in beat_times:
                    time_since_beat = t - bt
                    if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                        if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                            z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                            if time_since_beat < (1 / fps):
                                last_beat_time = bt
                        break

                if ENABLE_SPIRAL:
                    for dt in drop_times:
                        if abs(t - dt) < (1 / fps):
                            if dt != interact_time:  # don't spiral on interact drop
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
            else:
                # Interaction — spiral toward each other
                t_since_interact = t - interact_time
                ix, iy, iz = get_interaction_offset(t_since_interact, drone_id=1)
                x += ix
                y += iy
                z += iz
                # During interaction expand z range to allow meeting
                z_min, z_max = Z_MIN, Z_MAX

        # Landing approach
        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = TAKEOFF_HEIGHT * (1 - land_progress) + Z_MIN * land_progress
            z_min, z_max = Z_MIN, Z_MAX

        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z

    return x_pos, y_pos, z_pos

def get_drone2_positions(features, duration=111.0, fps=30, start_time=55.5):
    total_frames = int(duration * fps)
    start_frame = int(start_time * fps)
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    treble_energy = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= start_time) & (beat_times <= duration)]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= start_time) & (drop_times <= duration)]
    section_times = features['section_times']
    s1_end = section_times[1]
    s2_end = section_times[2]

    rms_smooth = smooth(rms, window=60)
    bass_smooth = smooth(bass_energy, window=60)
    treble_smooth = smooth(treble_energy, window=60)

    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6

    bass_norm = (bass_smooth - bass_smooth.min()) / (bass_smooth.max() - bass_smooth.min())
    bass_norm = 0.5 + bass_norm * 1.0

    treble_norm = (treble_smooth - treble_smooth.min()) / (treble_smooth.max() - treble_smooth.min())
    treble_norm = 0.0 + treble_norm * 0.2

    # Find same biggest drop for synchronized interaction
    onset_strength = features.get('onset_strength', rms_smooth)
    all_drops = features['drop_times']
    interact_time = find_biggest_drop(all_drops, onset_strength, rms_times, s1_end, s2_end)

    x_pos = np.zeros(total_frames)
    y_pos = np.zeros(total_frames)
    z_pos = np.zeros(total_frames)

    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN
    land_start = duration - 3.0

    for frame in range(start_frame, total_frames):
        t = frame / fps
        speed = get_value(t, rms_times, rms_norm)
        size = get_value(t, freq_times, bass_norm)
        treble_height = get_value(t, freq_times, treble_norm)

        in_interaction = (t >= interact_time and
                         t <= interact_time + INTERACT_DURATION)

        # Opposite direction figure-8 in lower zone
        angle -= speed * (2 * np.pi / fps) * 0.032
        x, y = figure8(angle, size)
        z = 0.9 + treble_height
        z_min, z_max = D2_S2_Z_MIN, D2_S2_Z_MAX

        # Fade in
        fade_in = min((t - start_time) / BLEND_DURATION, 1.0)
        x *= fade_in
        y *= fade_in

        if not in_interaction:
            # Beat pulse
            for bt in beat_times:
                time_since_beat = t - bt
                if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                    if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                        z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT * 0.8)
                        if time_since_beat < (1 / fps):
                            last_beat_time = bt
                    break

            # Spiral scaled for lower zone
            if ENABLE_SPIRAL:
                for dt in drop_times:
                    if abs(t - dt) < (1 / fps):
                        if dt != interact_time:
                            if (dt - last_spiral_time) >= SPIRAL_COOLDOWN:
                                current_spiral_start = t
                                last_spiral_time = t
                        break
                t_since_spiral = t - current_spiral_start
                if 0 <= t_since_spiral <= SPIRAL_DURATION:
                    sx, sy, sz = get_spiral_offset(t_since_spiral)
                    x += sx * 0.5
                    y += sy * 0.5
                    z += sz * 0.3

        else:
            # Interaction — spiral toward drone 1
            t_since_interact = t - interact_time
            ix, iy, iz = get_interaction_offset(t_since_interact, drone_id=2)
            x += ix
            y += iy
            z += iz
            z_min, z_max = Z_MIN, Z_MAX

        # Landing
        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = 0.9 * (1 - land_progress) + Z_MIN * land_progress
            z_min, z_max = Z_MIN, Z_MAX

        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z

    return x_pos, y_pos, z_pos, start_frame