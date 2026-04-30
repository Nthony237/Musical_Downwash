import numpy as np
import sys
from scipy.ndimage import uniform_filter1d
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
TAKEOFF_HEIGHT = 1.0
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.40
BEAT_PULSE_DURATION = 0.8
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 6.0
SPIRAL_HEIGHT = 0.8
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 0.9   # tighter for 3x2 structure
ENABLE_SPIRAL = True
BLEND_DURATION = 3.0
MIN_DRONE_DISTANCE = 0.3
SPEED_SCALE = 0.75

# ---- FIXED START/LAND POSITIONS ----
# Tighter triangle for 3x2m structure
D1_START = np.array([0.0,   0.3,  0.0])
D2_START = np.array([-0.4, -0.3,  0.0])
D3_START = np.array([0.4,  -0.3,  0.0])
D1_LAND  = np.array([0.0,   0.3,  0.0])
D2_LAND  = np.array([-0.4, -0.3,  0.0])
D3_LAND  = np.array([0.4,  -0.3,  0.0])

# ---- BOUNDS — tighter for 3x2 structure ----
X_MIN, X_MAX = -1.2, 1.2   # 3m wide / 2 with buffer
Y_MIN, Y_MAX = -0.8, 0.8   # 2m deep / 2 with buffer
Z_MIN, Z_MAX = 0.5, 2.2

# ---- HEIGHT ZONES ----
D1_Z_MIN, D1_Z_MAX = 0.5, 2.2
D2_Z_MIN, D2_Z_MAX = 0.5, 1.6
D3_Z_MIN, D3_Z_MAX = 0.5, 1.9

# ---- SECTION TIMESTAMPS ----
S1_END   = 55.0
S2_END   = 111.0
SONG_END = 166.6

# ---- CHOREOGRAPHY TIMESTAMPS ----
CIRCLE1_START      = 15.0
CIRCLE1_END        = 24.0
CIRCLE2_START      = 40.0
CIRCLE2_END        = 50.0
TORNADO_START      = 50.0
TORNADO_END        = 55.0
ARC_START          = 55.0
BOTH_CIRCLE_START  = 64.0
BOTH_CIRCLE_END    = 79.0
SWAY_START         = 79.0
SWAY_END           = 90.0
D2_SPIRAL_START    = 90.0
D2_SPIRAL_END      = 111.0
UPDOWN_START       = 116.0
UPDOWN_END         = 131.0
FINAL_CIRCLE_START = 131.0

# ---- LINE FORMATION TIMESTAMPS ----
LINE_START = 70.0   # during both circle section all form line briefly
LINE_END   = 79.0

# ---- RGB COLOR PALETTES per section ----
# (r, g, b) each 0-255
# Section 1 — warm white/gold, pulses blue on beat
S1_BASE_COLOR  = (255, 200, 100)
S1_BEAT_COLOR  = (100, 150, 255)

# Section 2 — cool purple/magenta, pulses white on beat
S2_BASE_COLOR  = (180,  50, 255)
S2_BEAT_COLOR  = (255, 255, 255)

# Section 3 — teal/cyan, pulses orange on beat
S3_BASE_COLOR  = (0,   220, 200)
S3_BEAT_COLOR  = (255, 120,  0)

def lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors."""
    t = np.clip(t, 0.0, 1.0)
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )

def get_color(t, beat_times, last_beat_t):
    """
    Returns (r, g, b) based on section and beat proximity.
    Pulses to beat color on each beat then fades back.
    """
    # Determine section
    if t <= S1_END:
        base, beat_col = S1_BASE_COLOR, S1_BEAT_COLOR
    elif t <= S2_END:
        base, beat_col = S2_BASE_COLOR, S2_BEAT_COLOR
    else:
        base, beat_col = S3_BASE_COLOR, S3_BEAT_COLOR

    # Find time since last beat
    past_beats = beat_times[beat_times <= t]
    if len(past_beats) == 0:
        return base

    time_since_beat = t - past_beats[-1]
    pulse_duration = 0.4   # how long the color pulse lasts

    if time_since_beat <= pulse_duration:
        # Pulse to beat color then fade back
        pulse_progress = time_since_beat / pulse_duration
        return lerp_color(beat_col, base, pulse_progress)
    else:
        return base

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

def get_beat_pulse(t_since_beat, duration=0.8, height=0.40):
    if t_since_beat < 0 or t_since_beat > duration:
        return 0.0
    return height * np.sin(np.pi * t_since_beat / duration)

def get_spiral_offset(t_since_drop):
    if t_since_drop < 0 or t_since_drop > SPIRAL_DURATION:
        return 0.0, 0.0, 0.0
    p = t_since_drop / SPIRAL_DURATION
    angle = p * 8 * np.pi
    r = SPIRAL_RADIUS_START + (SPIRAL_RADIUS_END - SPIRAL_RADIUS_START) * p
    return r * np.cos(angle), r * np.sin(angle), SPIRAL_HEIGHT * p

def tornado(t, start_time, duration=5.0, max_radius=0.9, max_height=1.5):
    t_since = t - start_time
    if t_since < 0 or t_since > duration:
        return 0.0, 0.0, 0.0
    p = t_since / duration
    angle = p * 4 * np.pi * SPEED_SCALE
    return max_radius * p * np.cos(angle), max_radius * p * np.sin(angle), max_height * p

def helix_spiral(t, start_time, duration=21.0, max_radius=0.8, max_height=1.0):
    t_since = t - start_time
    if t_since < 0 or t_since > duration:
        return 0.0, 0.0, 0.0
    p = t_since / duration
    angle = p * 6 * np.pi * SPEED_SCALE
    return max_radius * p * np.cos(angle), max_radius * p * np.sin(angle), max_height * p

def updown_alternating(t, drone_id, amplitude=0.35):
    phase = 0.0 if drone_id in [1, 2] else np.pi
    return amplitude * np.sin(2 * np.pi * t / 3.2 + phase)

def drift_hop(t, drone_id, drift_speed=0.08):
    """
    Slow drift across space while doing beat hops.
    Each drone drifts in a slightly different direction.
    """
    offsets = {1: 0.0, 2: 2 * np.pi / 3, 3: 4 * np.pi / 3}
    phase = offsets[drone_id]
    drift_angle = drift_speed * SPEED_SCALE * t + phase
    # Keep drift within tighter bounds
    x_drift = 0.6 * np.cos(drift_angle)
    y_drift = 0.4 * np.sin(drift_angle)
    return x_drift, y_drift

def line_formation(t, drone_id, formation_start):
    """
    All three drones form a horizontal line and move together.
    They sweep from one side to the other as a unit.
    drone 1 = center, drone 2 = left, drone 3 = right
    """
    t_since = t - formation_start
    duration = LINE_END - LINE_START

    # Fade in and out
    if t_since < 1.0:
        fade = t_since / 1.0
    elif t_since > duration - 1.0:
        fade = (duration - t_since) / 1.0
    else:
        fade = 1.0
    fade = np.clip(fade, 0, 1)

    # Sweep motion — all move together on X axis
    sweep = 0.6 * np.sin(np.pi * t_since / duration)

    # Line spacing on Y axis
    spacing = {1: 0.0, 2: -0.35, 3: 0.35}
    y_offset = spacing[drone_id]

    return sweep * fade, y_offset * fade

def smooth_fade(t, start, duration=2.0):
    t_since = t - start
    if t_since <= 0: return 0.0
    if t_since >= duration: return 1.0
    return np.sin(np.pi * t_since / (2 * duration)) ** 2

def blend(a, b, factor):
    return a * (1 - factor) + b * factor

def apply_beat_pulse(t, beat_times, last_beat, z, height, fps):
    for bt in beat_times:
        tsb = t - bt
        if 0 <= tsb <= BEAT_PULSE_DURATION:
            if (bt - last_beat) >= BEAT_COOLDOWN or bt == last_beat:
                z += get_beat_pulse(tsb, BEAT_PULSE_DURATION, height)
                if tsb < (1/fps):
                    last_beat = bt
            break
    return z, last_beat

def apply_collision_avoidance(positions_list, min_dist=MIN_DRONE_DISTANCE):
    n_drones = len(positions_list)
    total_frames = len(positions_list[0][0])
    corrected = [[p[0].copy(), p[1].copy(), p[2].copy()] for p in positions_list]

    for frame in range(total_frames):
        pts = [np.array([corrected[d][0][frame],
                         corrected[d][1][frame],
                         corrected[d][2][frame]]) for d in range(n_drones)]

        active = [i for i in range(n_drones)
                  if not (pts[i][0] == 0 and pts[i][1] == 0 and pts[i][2] == 0)]
        if len(active) < 2:
            continue

        for _ in range(10):
            any_fix = False
            for i in range(len(active)):
                for j in range(i+1, len(active)):
                    di, dj = active[i], active[j]
                    delta = pts[dj] - pts[di]
                    dist = np.linalg.norm(delta)
                    if dist < min_dist and dist > 0.001:
                        push = (min_dist - dist) / 2.0
                        direction = delta / dist
                        pts[di] -= direction * push
                        pts[dj] += direction * push
                        any_fix = True
            if not any_fix:
                break

        for d in range(n_drones):
            corrected[d][0][frame] = pts[d][0]
            corrected[d][1][frame] = pts[d][1]
            corrected[d][2][frame] = pts[d][2]

    return corrected

def find_biggest_drop(drop_times, onset_strength, rms_times, t_start, t_end):
    mask = (drop_times >= t_start) & (drop_times <= t_end)
    drops = drop_times[mask]
    if len(drops) == 0:
        return (t_start + t_end) / 2
    best_t, best_s = drops[0], 0
    for dt in drops:
        idx = min(np.searchsorted(rms_times, dt), len(onset_strength) - 1)
        if onset_strength[idx] > best_s:
            best_s = onset_strength[idx]
            best_t = dt
    return best_t

def get_drone1_positions(features, duration=SONG_END, fps=30):
    total_frames = int(duration * fps)
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times'][features['beat_times'] <= duration]
    drop_times = features['drop_times'][features['drop_times'] <= duration]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.6
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 0.8
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.25
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    x_pos  = np.full(total_frames, D1_START[0])
    y_pos  = np.full(total_frames, D1_START[1])
    z_pos  = np.full(total_frames, TAKEOFF_HEIGHT)
    colors = []

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    circle1_angle = 0.0
    circle2_angle = 0.0
    s2_angle      = 0.0
    orbit_angle   = 0.0
    final_angle   = 0.0
    land_start    = duration - 4.0
    prev_x = D1_START[0]
    prev_y = D1_START[1]

    for frame in range(total_frames):
        t     = frame / fps
        speed = get_value(t, rms_times, rms_norm)
        size  = get_value(t, freq_times, bass_norm)
        th    = get_value(t, freq_times, treb_norm)
        vocal = get_value(t, rms_times, rms_long_norm)

        x = D1_START[0]
        y = D1_START[1]
        z = TAKEOFF_HEIGHT + th
        z_min, z_max = D1_Z_MIN, D1_Z_MAX

        # =================== SECTION 1 ===================
        if t <= S1_END:

            if t < CIRCLE1_START:
                # Beat hops with drift
                dx, dy = drift_hop(t, drone_id=1)
                x = D1_START[0] + dx
                y = D1_START[1] + dy
                hop = BEAT_PULSE_HEIGHT * (0.3 + vocal * 0.7)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z, hop, fps)

            elif t <= CIRCLE1_END:
                circle1_angle += speed * SPEED_SCALE * 0.05
                radius = 0.5 + vocal * 0.4
                x = D1_START[0] + radius * np.cos(circle1_angle)
                y = D1_START[1] + radius * np.sin(circle1_angle)

            elif t < CIRCLE2_START:
                # BPM + spirals + drift
                dx, dy = drift_hop(t, drone_id=1)
                x = D1_START[0] + dx * 0.5
                y = D1_START[1] + dy * 0.5
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, fps)
                if ENABLE_SPIRAL:
                    for dt in drop_times:
                        if abs(t - dt) < (1/fps):
                            if (dt - last_spiral) >= SPIRAL_COOLDOWN:
                                spiral_start = t
                                last_spiral = t
                            break
                    tss = t - spiral_start
                    if 0 <= tss <= SPIRAL_DURATION:
                        sx, sy, sz = get_spiral_offset(tss)
                        x += sx; y += sy; z += sz

            elif t <= CIRCLE2_END:
                circle2_angle += speed * SPEED_SCALE * 0.04
                radius = 0.4 + vocal * 0.3
                x = D1_START[0] + radius * np.cos(circle2_angle)
                y = D1_START[1] + radius * np.sin(circle2_angle)
                z = 0.7 + th

            elif t <= TORNADO_END:
                tx, ty, tz = tornado(t, TORNADO_START,
                                     duration=TORNADO_END - TORNADO_START)
                fade = smooth_fade(t, TORNADO_START, duration=1.0)
                x = D1_START[0] + tx * fade
                y = D1_START[1] + ty * fade
                z = 0.5 + tz

            else:
                dx, dy = drift_hop(t, drone_id=1)
                x = D1_START[0] + dx * 0.3
                y = D1_START[1] + dy * 0.3

        # =================== SECTION 2 ===================
        elif t <= S2_END:
            z = 1.8 + th
            z_min, z_max = 1.5, 2.2

            # Line formation overrides during LINE_START to LINE_END
            if LINE_START <= t <= LINE_END:
                lx, ly = line_formation(t, drone_id=1,
                                        formation_start=LINE_START)
                x = lx
                y = ly
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, fps)

            elif t < BOTH_CIRCLE_START:
                s2_angle += speed * SPEED_SCALE * 0.035
                x = D1_START[0] + 1.0 * np.sin(s2_angle)
                y = D1_START[1] + 0.7 * np.sin(2 * s2_angle + np.pi / 4)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, fps)

            elif t <= BOTH_CIRCLE_END:
                orbit_angle += 1.0 * SPEED_SCALE * (1/fps)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                x = D1_START[0] + 0.9 * np.cos(orbit_angle) * fade
                y = D1_START[1] + 0.6 * np.sin(orbit_angle) * fade

            elif t <= SWAY_END:
                beat_idx = int(np.searchsorted(beat_times, t)) % 8
                sway_dir = 1.0 if beat_idx < 4 else -1.0
                fade = smooth_fade(t, SWAY_START, duration=2.0)
                x = D1_START[0] + sway_dir * 0.9 * vocal * fade
                y = D1_START[1]
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, fps)

            else:
                # Wide orbit while drone 2 spirals
                orbit_angle += 0.5 * SPEED_SCALE * (1/fps)
                x = D1_START[0] + 0.9 * np.cos(orbit_angle)
                y = D1_START[1] + 0.6 * np.sin(orbit_angle)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 1.2, fps)

        # =================== SECTION 3 ===================
        else:
            z_min, z_max = D1_Z_MIN, D1_Z_MAX
            fade = smooth_fade(t, S2_END, duration=BLEND_DURATION)

            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 10.0
                ox = 0.5 * np.cos(orbit_s3 * (t - S2_END))
                oy = 0.35 * np.sin(orbit_s3 * (t - S2_END))
                # Add drift during up/down
                dx, dy = drift_hop(t, drone_id=1, drift_speed=0.05)
                x = D1_START[0] + ox * fade + dx * 0.2 * fade
                y = D1_START[1] + oy * fade + dy * 0.2 * fade
                z = 2.0 + th + updown_alternating(t, drone_id=1,
                                                   amplitude=0.3) * fade

            else:
                final_angle += 2 * np.pi / 10.0 * (1/fps)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                radius = 0.9 if progress < 0.7 else 0.9 * (1 - (progress - 0.7) / 0.3)
                x = radius * np.cos(final_angle)
                y = radius * np.sin(final_angle) * 0.6  # ellipse fits 3x2
                z = 2.0 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, fps)

        # Boundary transitions
        if abs(t - S1_END) < 2.0 and t > S1_END:
            bf = smooth_fade(t, S1_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)
        if abs(t - S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)

        # Landing
        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, D1_LAND[0], lp)
            y = blend(y, D1_LAND[1], lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y

        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z
        colors.append(get_color(t, beat_times, last_beat))

    return x_pos, y_pos, z_pos, colors


def get_drone2_positions(features, duration=SONG_END, fps=30, start_time=S1_END):
    total_frames = int(duration * fps)
    start_frame  = int(start_time * fps)
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times_full = beat_times[beat_times <= duration]
    beat_times = beat_times[(beat_times >= start_time) & (beat_times <= duration)]
    drop_times = features['drop_times'][features['drop_times'] <= duration]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.6
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 0.8
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.25
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    x_pos  = np.zeros(total_frames)
    y_pos  = np.zeros(total_frames)
    z_pos  = np.zeros(total_frames)
    colors = [S1_BASE_COLOR] * total_frames

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    orbit_angle  = np.pi
    final_angle  = 0.0
    land_start   = duration - 4.0
    prev_x = D2_START[0]
    prev_y = D2_START[1]

    for frame in range(start_frame, total_frames):
        t     = frame / fps
        speed = get_value(t, rms_times, rms_norm)
        size  = get_value(t, freq_times, bass_norm)
        th    = get_value(t, freq_times, treb_norm)
        vocal = get_value(t, rms_times, rms_long_norm)

        x = D2_START[0]
        y = D2_START[1]
        z = 0.9 + th
        z_min, z_max = D2_Z_MIN, D2_Z_MAX
        fade_in = smooth_fade(t, start_time, duration=BLEND_DURATION)

        # =================== SECTION 2 ===================
        if t <= S2_END:

            # Line formation
            if LINE_START <= t <= LINE_END:
                lx, ly = line_formation(t, drone_id=2,
                                        formation_start=LINE_START)
                x = lx
                y = ly
                z = 1.5 + th   # drone 2 in line at middle height
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 0.8, fps)

            elif t < BOTH_CIRCLE_START:
                t_since = t - ARC_START
                bx = 0.7 * np.cos(1.5 * SPEED_SCALE * t_since)
                by = 0.6 * np.sin(1.5 * SPEED_SCALE * t_since)
                x = D2_START[0] + bx * fade_in
                y = D2_START[1] + by * fade_in

            elif t <= BOTH_CIRCLE_END:
                orbit_angle += 0.6 * SPEED_SCALE * (1/fps)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                x = D2_START[0] + 0.9 * np.cos(orbit_angle) * fade
                y = D2_START[1] + 0.6 * np.sin(orbit_angle) * fade

            elif t <= SWAY_END:
                beat_idx = int(np.searchsorted(beat_times, t)) % 8
                sway_dir = 1.0 if beat_idx < 4 else -1.0
                fade = smooth_fade(t, SWAY_START, duration=2.0)
                x = D2_START[0] + sway_dir * 0.9 * vocal * fade
                y = D2_START[1]
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 0.8, fps)

            else:
                helix_dur = D2_SPIRAL_END - D2_SPIRAL_START
                descent_start = D2_SPIRAL_END - 5.0

                if t < descent_start:
                    hx, hy, hz = helix_spiral(t, D2_SPIRAL_START,
                                              duration=helix_dur,
                                              max_radius=0.8, max_height=0.9)
                    x = D2_START[0] + hx
                    y = D2_START[1] + hy
                    z = 0.8 + hz + th
                    z_min, z_max = D2_Z_MIN, 2.0
                else:
                    dp = (t - descent_start) / 5.0
                    hx, hy, hz = helix_spiral(descent_start, D2_SPIRAL_START,
                                              duration=helix_dur,
                                              max_radius=0.8, max_height=0.9)
                    peak_x = D2_START[0] + hx
                    peak_y = D2_START[1] + hy
                    peak_z = 0.8 + hz + th
                    x = blend(peak_x, D2_START[0], dp)
                    y = blend(peak_y, D2_START[1], dp)
                    z = blend(peak_z, 0.8, dp)
                    z_min, z_max = D2_Z_MIN, 2.0

        # =================== SECTION 3 ===================
        else:
            z_min, z_max = Z_MIN, Z_MAX
            fade = smooth_fade(t, S2_END + 2.0, duration=BLEND_DURATION)

            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 10.0
                base_angle = 2 * np.pi / 3
                ox = 0.5 * np.cos(orbit_s3 * (t - S2_END) + base_angle)
                oy = 0.35 * np.sin(orbit_s3 * (t - S2_END) + base_angle)
                dx, dy = drift_hop(t, drone_id=2, drift_speed=0.05)
                x = D2_START[0] + ox * fade + dx * 0.15 * fade
                y = D2_START[1] + oy * fade + dy * 0.15 * fade
                z = 1.3 + th + updown_alternating(t, drone_id=2,
                                                   amplitude=0.3) * fade

            else:
                final_angle += 2 * np.pi / 10.0 * (1/fps)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                radius = 0.65 if progress < 0.7 else 0.65 * (1 - (progress - 0.7) / 0.3)
                delay_angle = final_angle - (1.5 * 2 * np.pi / 10.0)
                x = radius * np.cos(delay_angle)
                y = radius * np.sin(delay_angle) * 0.6
                z = 1.3 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 0.8, fps)

        if abs(t - S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)

        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, D2_LAND[0], lp)
            y = blend(y, D2_LAND[1], lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y

        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z
        colors[frame] = get_color(t, beat_times_full, last_beat)

    return x_pos, y_pos, z_pos, start_frame, colors


def get_drone3_positions(features, duration=SONG_END, fps=30, start_time=S2_END):
    total_frames = int(duration * fps)
    start_frame  = int(start_time * fps)
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times_full = beat_times[beat_times <= duration]
    beat_times = beat_times[(beat_times >= start_time) & (beat_times <= duration)]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= start_time) & (drop_times <= duration)]

    rms_s    = smooth(rms,    window=60)
    treb_s   = smooth(treble, window=60)

    rms_norm  = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.6
    treb_norm = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.25

    x_pos  = np.zeros(total_frames)
    y_pos  = np.zeros(total_frames)
    z_pos  = np.zeros(total_frames)
    colors = [S1_BASE_COLOR] * total_frames

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    final_angle  = 0.0
    land_start   = duration - 4.0

    for frame in range(start_frame, total_frames):
        t  = frame / fps
        th = get_value(t, rms_times, treb_norm)

        fade_in = smooth_fade(t, start_time + 1.0, duration=BLEND_DURATION)

        x = D3_START[0]
        y = D3_START[1]
        z = 0.8 + th
        z_min, z_max = D3_Z_MIN, D3_Z_MAX

        if t <= UPDOWN_END:
            orbit_s3 = 2 * np.pi / 10.0
            base_angle = 4 * np.pi / 3
            ox = 0.5 * np.cos(orbit_s3 * (t - S2_END) + base_angle)
            oy = 0.35 * np.sin(orbit_s3 * (t - S2_END) + base_angle)
            dx, dy = drift_hop(t, drone_id=3, drift_speed=0.05)
            x = D3_START[0] + ox * fade_in + dx * 0.15 * fade_in
            y = D3_START[1] + oy * fade_in + dy * 0.15 * fade_in
            z = 0.8 + th + updown_alternating(t, drone_id=3,
                                               amplitude=0.3) * fade_in

        else:
            final_angle += 2 * np.pi / 10.0 * (1/fps)
            progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
            radius = 0.4 if progress < 0.7 else 0.4 * (1 - (progress - 0.7) / 0.3)
            delay_angle = final_angle - (3.0 * 2 * np.pi / 10.0)
            x = radius * np.cos(delay_angle) * fade_in
            y = radius * np.sin(delay_angle) * 0.6 * fade_in
            z = 0.8 + th
            z_min, z_max = Z_MIN, Z_MAX

            z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                            BEAT_PULSE_HEIGHT * 0.7, fps)

            if ENABLE_SPIRAL:
                for dt in drop_times:
                    if abs(t - dt) < (1/fps):
                        if (dt - last_spiral) >= SPIRAL_COOLDOWN:
                            spiral_start = t
                            last_spiral = t
                        break
                tss = t - spiral_start
                if 0 <= tss <= SPIRAL_DURATION:
                    sx, sy, sz = get_spiral_offset(tss)
                    x += sx * 0.3 * fade_in
                    y += sy * 0.2 * fade_in
                    z += sz * 0.3

        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, D3_LAND[0], lp)
            y = blend(y, D3_LAND[1], lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z
        colors[frame] = get_color(t, beat_times_full, last_beat)

    return x_pos, y_pos, z_pos, start_frame, colors