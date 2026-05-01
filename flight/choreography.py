import numpy as np
import sys
from scipy.ndimage import uniform_filter1d
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
TAKEOFF_HEIGHT = 1.0
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.45   # more noticeable hops
BEAT_PULSE_DURATION = 0.8
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 6.0
SPIRAL_HEIGHT = 0.9
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 1.2
ENABLE_SPIRAL = True
BLEND_DURATION = 3.0
MIN_DRONE_DISTANCE = 0.3
SPEED_SCALE = 0.85   # slightly faster for more upbeat feel

# ---- FABRIC CENTER (downwash target) ----
FABRIC_CENTER_X = 0.0
FABRIC_CENTER_Y = 0.0
FABRIC_X_HALF = 0.5   # ±0.5m on X
FABRIC_Y_HALF = 1.0   # ±1.0m on Y

# ---- START/LAND POSITIONS ----
D1_START = np.array([0.0,   1.0,  0.0])
D2_START = np.array([-1.0, -1.0,  0.0])
D3_START = np.array([1.0,  -1.0,  0.0])

# ---- BOUNDS ----
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
Z_MIN, Z_MAX = 0.5, 2.5

# ---- HEIGHT ZONES ----
D1_Z_MIN, D1_Z_MAX = 0.5, 2.5
D2_Z_MIN, D2_Z_MAX = 0.5, 1.6
D3_Z_MIN, D3_Z_MAX = 0.5, 1.9

# ---- SECTION TIMESTAMPS ----
S1_END   = 55.0    # 0:55 drone 2 enters
S2_END   = 111.0   # 1:51 drone 3 enters
SONG_END = 166.6

# ---- CHOREOGRAPHY TIMESTAMPS ----
# Section 1
BPM1_START    = 0.0
CIRCLE1_START = 15.0
CIRCLE1_END   = 24.0
BPM2_START    = 24.0
CIRCLE2_START = 40.0
CIRCLE2_END   = 50.0
TORNADO_START = 50.0
TORNADO_END   = 55.0

# Section 2
ARC_START         = 55.0
ARC_END           = 64.0    # 1:04
BOTH_CIRCLE_START = 64.0
BOTH_CIRCLE_END   = 79.0    # 1:19
SWAY_START        = 79.0
SWAY_END          = 90.0    # 1:30
D1_ORBIT_START    = 90.0
D2_HELIX_START    = 90.0
D2_HELIX_END      = 111.0

# Section 3
UPDOWN_START       = 111.0   # 1:51
UPDOWN_END         = 131.0   # 2:11
FINAL_CIRCLE_START = 131.0
FINAL_CIRCLE_END   = SONG_END

# ---- RGB ----
S1_BASE_COLOR = (255, 200, 80)
S1_BEAT_COLOR = (80,  150, 255)
S2_BASE_COLOR = (200,  40, 255)
S2_BEAT_COLOR = (255, 255, 255)
S3_BASE_COLOR = (0,   220, 180)
S3_BEAT_COLOR = (255, 100,   0)

def lerp_color(c1, c2, t):
    t = np.clip(t, 0.0, 1.0)
    return (int(c1[0]+(c2[0]-c1[0])*t),
            int(c1[1]+(c2[1]-c1[1])*t),
            int(c1[2]+(c2[2]-c1[2])*t))

def get_color(t, beat_times):
    if t <= S1_END:
        base, beat_col = S1_BASE_COLOR, S1_BEAT_COLOR
    elif t <= S2_END:
        base, beat_col = S2_BASE_COLOR, S2_BEAT_COLOR
    else:
        base, beat_col = S3_BASE_COLOR, S3_BEAT_COLOR
    past = beat_times[beat_times <= t]
    if len(past) == 0:
        return base
    tsb = t - past[-1]
    if tsb <= 0.4:
        return lerp_color(beat_col, base, tsb / 0.4)
    return base

def clamp(x, y, z, z_min=Z_MIN, z_max=Z_MAX):
    return (np.clip(x, X_MIN, X_MAX),
            np.clip(y, Y_MIN, Y_MAX),
            np.clip(z, z_min, z_max))

def smooth(signal, window=60):
    return uniform_filter1d(signal.astype(float), size=window)

def get_value(t, times, values):
    idx = np.searchsorted(times, t)
    return values[min(idx, len(values)-1)]

def get_beat_pulse(tsb, duration=0.8, height=0.45):
    if tsb < 0 or tsb > duration:
        return 0.0
    return height * np.sin(np.pi * tsb / duration)

def get_spiral_offset(tss):
    if tss < 0 or tss > SPIRAL_DURATION:
        return 0.0, 0.0, 0.0
    p = tss / SPIRAL_DURATION
    angle = p * 8 * np.pi
    r = SPIRAL_RADIUS_START + (SPIRAL_RADIUS_END - SPIRAL_RADIUS_START) * p
    return r*np.cos(angle), r*np.sin(angle), SPIRAL_HEIGHT*p

def tornado_move(t, start, duration=5.0, max_r=1.1, max_h=1.5):
    ts = t - start
    if ts < 0 or ts > duration:
        return 0.0, 0.0, 0.0
    p = ts / duration
    angle = p * 4 * np.pi * SPEED_SCALE
    return max_r*p*np.cos(angle), max_r*p*np.sin(angle), max_h*p

def helix_move(t, start, duration=21.0, max_r=1.0, max_h=1.0):
    ts = t - start
    if ts < 0 or ts > duration:
        return 0.0, 0.0, 0.0
    p = ts / duration
    angle = p * 6 * np.pi * SPEED_SCALE
    return max_r*p*np.cos(angle), max_r*p*np.sin(angle), max_h*p

def updown_alt(t, drone_id, amplitude=0.4):
    phase = 0.0 if drone_id in [1, 2] else np.pi
    return amplitude * np.sin(2 * np.pi * t / 3.2 + phase)

def smooth_fade(t, start, duration=2.0):
    ts = t - start
    if ts <= 0: return 0.0
    if ts >= duration: return 1.0
    return np.sin(np.pi * ts / (2 * duration)) ** 2

def blend(a, b, f):
    return a * (1-f) + b * f

def fabric_drift(t, drone_id, drift_speed=0.06):
    """
    Drift that periodically pulls toward fabric center [0,0]
    then back to start area. Creates downwash passes over fabric.
    Each drone has different phase so they take turns passing over.
    """
    offsets = {1: 0.0, 2: 2*np.pi/3, 3: 4*np.pi/3}
    phase = offsets[drone_id]
    # Oscillate between 0 (at center) and 1 (at start offset)
    pull = (np.sin(drift_speed * SPEED_SCALE * t + phase) + 1) / 2
    # Start offsets relative to fabric center
    start_offsets = {
        1: (D1_START[0] - FABRIC_CENTER_X, D1_START[1] - FABRIC_CENTER_Y),
        2: (D2_START[0] - FABRIC_CENTER_X, D2_START[1] - FABRIC_CENTER_Y),
        3: (D3_START[0] - FABRIC_CENTER_X, D3_START[1] - FABRIC_CENTER_Y)
    }
    sx, sy = start_offsets[drone_id]
    # Scale down so it stays reasonable
    x_drift = sx * pull * 0.4
    y_drift = sy * pull * 0.4
    return x_drift, y_drift

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
    n = len(positions_list)
    total = len(positions_list[0][0])
    corr = [[p[0].copy(), p[1].copy(), p[2].copy()] for p in positions_list]

    for frame in range(total):
        pts = [np.array([corr[d][0][frame], corr[d][1][frame], corr[d][2][frame]])
               for d in range(n)]
        active = [i for i in range(n)
                  if not (pts[i][0]==0 and pts[i][1]==0 and pts[i][2]==0)]
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
        for d in range(n):
            corr[d][0][frame] = pts[d][0]
            corr[d][1][frame] = pts[d][1]
            corr[d][2][frame] = pts[d][2]
    return corr

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

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.7
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    x_pos  = np.full(total_frames, D1_START[0])
    y_pos  = np.full(total_frames, D1_START[1])
    z_pos  = np.full(total_frames, TAKEOFF_HEIGHT)
    colors = []

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    c1_angle = 0.0
    c2_angle = 0.0
    s2_angle = 0.0
    orb_angle = 0.0
    fin_angle = 0.0
    land_start = duration - 4.0
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

        # ===== SECTION 1 =====
        if t <= S1_END:

            # 0:00-0:14 BPM hops drifting toward fabric center
            if t < CIRCLE1_START:
                dx, dy = fabric_drift(t, drone_id=1, drift_speed=0.07)
                x = FABRIC_CENTER_X + dx
                y = FABRIC_CENTER_Y + dy
                hop = BEAT_PULSE_HEIGHT * (0.4 + vocal * 0.6)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, hop, fps)

            # 0:15-0:24 Circle with lyrics — medium radius, energetic
            elif t <= CIRCLE1_END:
                c1_angle += speed * SPEED_SCALE * 0.07  # faster circle
                radius = 0.6 + vocal * 0.5
                x = D1_START[0] + radius * np.cos(c1_angle)
                y = D1_START[1] + radius * np.sin(c1_angle)
                # Small hops during circle for upbeat feel
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.5, fps)

            # 0:24-0:40 BPM + spirals, drifting toward fabric
            elif t < CIRCLE2_START:
                dx, dy = fabric_drift(t, drone_id=1, drift_speed=0.06)
                x = FABRIC_CENTER_X + dx
                y = FABRIC_CENTER_Y + dy
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, fps)
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

            # 0:40-0:50 Second circle, lower height
            elif t <= CIRCLE2_END:
                c2_angle += speed * SPEED_SCALE * 0.06
                radius = 0.5 + vocal * 0.4
                x = D1_START[0] + radius * np.cos(c2_angle)
                y = D1_START[1] + radius * np.sin(c2_angle)
                z = 0.8 + th   # lower than first circle
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.6, fps)

            # 0:50-0:55 Tornado rising up before drone 2 enters
            elif t <= TORNADO_END:
                tx, ty, tz = tornado_move(t, TORNADO_START,
                                          duration=TORNADO_END - TORNADO_START)
                fade = smooth_fade(t, TORNADO_START, duration=1.0)
                x = D1_START[0] + tx * fade
                y = D1_START[1] + ty * fade
                z = 0.5 + tz

            else:
                dx, dy = fabric_drift(t, drone_id=1)
                x = FABRIC_CENTER_X + dx * 0.5
                y = FABRIC_CENTER_Y + dy * 0.5

        # ===== SECTION 2 =====
        elif t <= S2_END:
            z = 1.8 + th
            z_min, z_max = 1.5, 2.5

            # 0:55-1:04 Wide lissajous while drone 2 bows
            if t < BOTH_CIRCLE_START:
                s2_angle += speed * SPEED_SCALE * 0.045
                x = D1_START[0] + 1.1 * np.sin(s2_angle)
                y = D1_START[1] + 1.0 * np.sin(2 * s2_angle + np.pi/4)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, fps)

            # 1:04-1:19 Both circle each other — drone 1 faster
            elif t <= BOTH_CIRCLE_END:
                orb_angle += 1.1 * SPEED_SCALE * (1/fps)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                # Drone 1 orbits at 0 degrees phase
                x = 1.0 * np.cos(orb_angle) * fade
                y = 1.0 * np.sin(orb_angle) * fade
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.7, fps)

            # 1:19-1:30 Both sway left/right on Y axis — energetic
            elif t <= SWAY_END:
                fade = smooth_fade(t, SWAY_START, duration=1.5)
                # Faster sway tied to beat
                beat_idx = int(np.searchsorted(beat_times, t)) % 4
                sway_dir = 1.0 if beat_idx < 2 else -1.0
                x = sway_dir * 1.0 * vocal * fade
                y = D1_START[1] * (1 - fade * 0.5)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 1.1, fps)

            # 1:30-1:51 Drone 1 wide orbit + strong beat hops
            else:
                orb_angle += 0.7 * SPEED_SCALE * (1/fps)
                x = 1.0 * np.cos(orb_angle)
                y = 0.8 * np.sin(orb_angle)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 1.4, fps)

        # ===== SECTION 3 =====
        else:
            z_min, z_max = D1_Z_MIN, D1_Z_MAX
            fade = smooth_fade(t, S2_END, duration=BLEND_DURATION)

            # 1:51-2:11 Up/down alternating with orbit over fabric
            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 8.0   # faster orbit
                ox = FABRIC_X_HALF * np.cos(orbit_s3 * (t - S2_END))
                oy = FABRIC_Y_HALF * np.sin(orbit_s3 * (t - S2_END))
                x = FABRIC_CENTER_X + ox * fade
                y = FABRIC_CENTER_Y + oy * fade
                z = 2.0 + th + updown_alt(t, drone_id=1, amplitude=0.35) * fade

            # 2:11-end Final circle expanding then contracting to land
            else:
                fin_angle += 2 * np.pi / 9.0 * (1/fps)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                if progress < 0.5:
                    radius = 0.4 + progress * 1.2   # expand
                else:
                    radius = 1.0 * (1 - (progress - 0.5) / 0.5)  # contract
                x = radius * np.cos(fin_angle)
                y = radius * np.sin(fin_angle)
                z = 1.8 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, fps)

        # Section boundary smoothing
        if abs(t - S1_END) < 2.0 and t > S1_END:
            bf = smooth_fade(t, S1_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)
        if abs(t - S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)

        # Landing — blend to 0,0 so initPos returns to start
        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, 0.0, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y
        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z
        colors.append(get_color(t, beat_times))

    return x_pos, y_pos, z_pos, colors


def get_drone2_positions(features, duration=SONG_END, fps=30, start_time=S1_END):
    total_frames = int(duration * fps)
    start_frame  = int(start_time * fps)
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times_full = features['beat_times'][features['beat_times'] <= duration]
    beat_times = beat_times_full[beat_times_full >= start_time]
    drop_times = features['drop_times'][features['drop_times'] <= duration]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.7
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    x_pos  = np.zeros(total_frames)
    y_pos  = np.zeros(total_frames)
    z_pos  = np.zeros(total_frames)
    colors = [S1_BASE_COLOR] * total_frames

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    orb_angle    = np.pi   # starts at 180 degrees opposite drone 1
    fin_angle    = 0.0
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

        # ===== SECTION 2 =====
        if t <= S2_END:

            # 0:55-1:04 Violin bow arc — full elliptical sweep
            if t < BOTH_CIRCLE_START:
                ts = t - ARC_START
                bx = 0.9 * np.cos(1.8 * SPEED_SCALE * ts)
                by = 1.1 * np.sin(1.8 * SPEED_SCALE * ts)
                x = D2_START[0] + bx * fade_in
                y = D2_START[1] + by * fade_in
                # Hops during bow arc for energy
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.7, fps)

            # 1:04-1:19 Both circle — drone 2 slower, opposite direction
            elif t <= BOTH_CIRCLE_END:
                orb_angle += 0.7 * SPEED_SCALE * (1/fps)  # slower
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                # Drone 2 at 180 degrees — opposite side of circle
                x = 1.0 * np.cos(orb_angle + np.pi) * fade
                y = 1.0 * np.sin(orb_angle + np.pi) * fade
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.7, fps)

            # 1:19-1:30 Sway together with drone 1
            elif t <= SWAY_END:
                fade = smooth_fade(t, SWAY_START, duration=1.5)
                beat_idx = int(np.searchsorted(beat_times, t)) % 4
                sway_dir = 1.0 if beat_idx < 2 else -1.0
                x = sway_dir * 1.0 * vocal * fade
                y = D2_START[1] * (1 - fade * 0.5)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.9, fps)

            # 1:30-1:51 Big ascending helix with descent at end
            else:
                helix_dur = D2_HELIX_END - D2_HELIX_START
                descent_start = D2_HELIX_END - 5.0
                if t < descent_start:
                    hx, hy, hz = helix_move(t, D2_HELIX_START,
                                            duration=helix_dur,
                                            max_r=1.0, max_h=1.1)
                    x = D2_START[0] + hx
                    y = D2_START[1] + hy
                    z = 0.8 + hz + th
                    z_min, z_max = D2_Z_MIN, 2.2
                else:
                    dp = (t - descent_start) / 5.0
                    hx, hy, hz = helix_move(descent_start, D2_HELIX_START,
                                            duration=helix_dur,
                                            max_r=1.0, max_h=1.1)
                    x = blend(D2_START[0]+hx, D2_START[0], dp)
                    y = blend(D2_START[1]+hy, D2_START[1], dp)
                    z = blend(0.8+hz+th, 0.8, dp)
                    z_min, z_max = D2_Z_MIN, 2.2

        # ===== SECTION 3 =====
        else:
            z_min, z_max = Z_MIN, Z_MAX
            fade = smooth_fade(t, S2_END + 2.0, duration=BLEND_DURATION)

            # 1:51-2:11 Up/down alternating + orbit over fabric
            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 8.0
                base_angle = 2 * np.pi / 3
                ox = FABRIC_X_HALF * np.cos(orbit_s3*(t-S2_END) + base_angle)
                oy = FABRIC_Y_HALF * np.sin(orbit_s3*(t-S2_END) + base_angle)
                dx, dy = fabric_drift(t, drone_id=2, drift_speed=0.04)
                x = FABRIC_CENTER_X + ox*fade + dx*0.2*fade
                y = FABRIC_CENTER_Y + oy*fade + dy*0.2*fade
                z = 1.3 + th + updown_alt(t, drone_id=2, amplitude=0.35) * fade

            # 2:11-end Final circle middle radius
            else:
                fin_angle += 2 * np.pi / 9.0 * (1/fps)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                if progress < 0.5:
                    radius = 0.3 + progress * 0.8
                else:
                    radius = 0.7 * (1 - (progress - 0.5) / 0.5)
                delay_angle = fin_angle - (1.5 * 2 * np.pi / 9.0)
                x = radius * np.cos(delay_angle)
                y = radius * np.sin(delay_angle)
                z = 1.3 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.8, fps)

        if abs(t - S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)

        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, 0.0, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y
        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z
        colors[frame] = get_color(t, beat_times_full)

    return x_pos, y_pos, z_pos, start_frame, colors


def get_drone3_positions(features, duration=SONG_END, fps=30, start_time=S2_END):
    total_frames = int(duration * fps)
    start_frame  = int(start_time * fps)
    rms_times  = features['rms_times']
    rms        = features['rms']
    treble     = features['treble_energy']
    beat_times_full = features['beat_times'][features['beat_times'] <= duration]
    beat_times = beat_times_full[beat_times_full >= start_time]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= start_time) & (drop_times <= duration)]

    rms_s    = smooth(rms,    window=60)
    treb_s   = smooth(treble, window=60)

    rms_norm  = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.7
    treb_norm = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.25

    x_pos  = np.zeros(total_frames)
    y_pos  = np.zeros(total_frames)
    z_pos  = np.zeros(total_frames)
    colors = [S1_BASE_COLOR] * total_frames

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    fin_angle    = 0.0
    land_start   = duration - 4.0

    for frame in range(start_frame, total_frames):
        t  = frame / fps
        th = get_value(t, rms_times, treb_norm)
        fade_in = smooth_fade(t, start_time + 1.0, duration=BLEND_DURATION)

        x = D3_START[0]
        y = D3_START[1]
        z = 0.8 + th
        z_min, z_max = D3_Z_MIN, D3_Z_MAX

        # 1:51-2:11 Up/down alternating OPPOSITE + orbit
        if t <= UPDOWN_END:
            orbit_s3 = 2 * np.pi / 8.0
            base_angle = 4 * np.pi / 3
            ox = FABRIC_X_HALF * np.cos(orbit_s3*(t-S2_END) + base_angle)
            oy = FABRIC_Y_HALF * np.sin(orbit_s3*(t-S2_END) + base_angle)
            dx, dy = fabric_drift(t, drone_id=3, drift_speed=0.04)
            x = FABRIC_CENTER_X + ox*fade_in + dx*0.2*fade_in
            y = FABRIC_CENTER_Y + oy*fade_in + dy*0.2*fade_in
            z = 0.8 + th + updown_alt(t, drone_id=3, amplitude=0.35) * fade_in

        # 2:11-end Final circle innermost, spreads then contracts
        else:
            fin_angle += 2 * np.pi / 9.0 * (1/fps)
            progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
            if progress < 0.5:
                radius = 0.2 + progress * 0.6
            else:
                radius = 0.5 * (1 - (progress - 0.5) / 0.5)
            delay_angle = fin_angle - (3.0 * 2 * np.pi / 9.0)
            x = radius * np.cos(delay_angle) * fade_in
            y = radius * np.sin(delay_angle) * fade_in
            z = 0.8 + th
            z_min, z_max = Z_MIN, Z_MAX

            z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                            z, BEAT_PULSE_HEIGHT * 0.7, fps)

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
                    y += sy * 0.3 * fade_in
                    z += sz * 0.3

        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, 0.0, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        x, y, z = clamp(x, y, z, z_min, z_max)
        x_pos[frame] = x
        y_pos[frame] = y
        z_pos[frame] = z
        colors[frame] = get_color(t, beat_times_full)

    return x_pos, y_pos, z_pos, start_frame, colors