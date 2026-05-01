import numpy as np
import time
from scipy.ndimage import uniform_filter1d
import sys
import os
import threading
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
SIM = False
DRY_RUN = False
Hz = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, 'audio', 'robots_mixdown.mp3')
TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 3.0
LAND_DURATION = 2.5
SONG_END = 166.6
SPEED_SCALE = 0.85

# ---- SAFETY BOUNDS ----
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
Z_MIN, Z_MAX = 0.5, 2.5

# ---- START POSITIONS ----
D1_START = np.array([0.0,   1.0,  0.0])
D2_START = np.array([-1.0, -1.0,  0.0])
D3_START = np.array([1.0,  -1.0,  0.0])

# ---- FABRIC CENTER (2m x 1m structure) ----
FABRIC_CENTER_X = 0.0
FABRIC_CENTER_Y = 0.0
FABRIC_X_HALF   = 0.5
FABRIC_Y_HALF   = 1.0

# ---- HEIGHT ZONES ----
D1_Z_MIN, D1_Z_MAX = 0.5, 2.5
D2_Z_MIN, D2_Z_MAX = 0.5, 1.6
D3_Z_MIN, D3_Z_MAX = 0.5, 1.9

# ---- BEAT SETTINGS ----
BEAT_COOLDOWN      = 1.6
BEAT_PULSE_HEIGHT  = 0.45
BEAT_PULSE_DURATION = 0.8

# ---- SPIRAL SETTINGS ----
SPIRAL_DURATION    = 4.0
SPIRAL_COOLDOWN    = 6.0
SPIRAL_HEIGHT      = 0.9
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END  = 1.2
ENABLE_SPIRAL      = True
BLEND_DURATION     = 3.0

# ---- SECTION TIMESTAMPS ----
S1_END   = 55.0
S2_END   = 111.0

# ---- CHOREOGRAPHY TIMESTAMPS ----
CIRCLE1_START     = 15.0
CIRCLE1_END       = 24.0
CIRCLE2_START     = 40.0
CIRCLE2_END       = 50.0
TORNADO_START     = 50.0
TORNADO_END       = 55.0
ARC_START         = 55.0
BOTH_CIRCLE_START = 64.0
BOTH_CIRCLE_END   = 79.0
SWAY_START        = 79.0
SWAY_END          = 90.0
D2_HELIX_START    = 90.0
D2_HELIX_END      = 111.0
UPDOWN_START      = 111.0
UPDOWN_END        = 131.0
FINAL_CIRCLE_START = 131.0

# ---- DRONE 2/3 TAKEOFF TIMING ----
DRONE2_TAKEOFF_DELAY    = 50.0
DRONE2_TAKEOFF_HEIGHT   = 0.8
DRONE2_TAKEOFF_DURATION = 3.0
DRONE3_TAKEOFF_DELAY    = 106.0
DRONE3_TAKEOFF_HEIGHT   = 0.8
DRONE3_TAKEOFF_DURATION = 3.0

# ---- RGB ----
S1_BASE_COLOR = (255, 200,  80)
S1_BEAT_COLOR = ( 80, 150, 255)
S2_BASE_COLOR = (200,  40, 255)
S2_BEAT_COLOR = (255, 255, 255)
S3_BASE_COLOR = (  0, 220, 180)
S3_BEAT_COLOR = (255, 100,   0)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

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


def send_color(cf, r, g, b):
    try:
        cf.setParam('ring.effect',     7)
        cf.setParam('ring.solidRed',   r)
        cf.setParam('ring.solidGreen', g)
        cf.setParam('ring.solidBlue',  b)
    except Exception:
        pass


def clamp_position(position, z_min=Z_MIN, z_max=Z_MAX):
    x = np.clip(position[0], X_MIN, X_MAX)
    y = np.clip(position[1], Y_MIN, Y_MAX)
    z = np.clip(position[2], z_min, z_max)
    return np.array([x, y, z])


def smooth(signal, window=60):
    from scipy.ndimage import uniform_filter1d
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


def fabric_drift(t, drone_id, drift_speed=0.06):
    """
    Drift that pulls toward fabric center then back out.
    Each drone has a different phase so they take turns
    passing over the fabric — maximising downwash visibility.
    While drifting the drone is always moving, never static.
    """
    offsets = {1: 0.0, 2: 2*np.pi/3, 3: 4*np.pi/3}
    phase = offsets[drone_id]
    # pull oscillates 0→1 (at center) → 0 (at offset)
    pull = (np.sin(drift_speed * SPEED_SCALE * t + phase) + 1) / 2
    start_offsets = {
        1: (D1_START[0] - FABRIC_CENTER_X, D1_START[1] - FABRIC_CENTER_Y),
        2: (D2_START[0] - FABRIC_CENTER_X, D2_START[1] - FABRIC_CENTER_Y),
        3: (D3_START[0] - FABRIC_CENTER_X, D3_START[1] - FABRIC_CENTER_Y),
    }
    sx, sy = start_offsets[drone_id]
    return sx * pull * 0.4, sy * pull * 0.4


def smooth_fade(t, start, duration=2.0):
    ts = t - start
    if ts <= 0: return 0.0
    if ts >= duration: return 1.0
    return np.sin(np.pi * ts / (2 * duration)) ** 2


def blend(a, b, f):
    return a * (1-f) + b * f


def apply_beat_pulse(t, beat_times, last_beat, z, height, hz):
    for bt in beat_times:
        tsb = t - bt
        if 0 <= tsb <= BEAT_PULSE_DURATION:
            if (bt - last_beat) >= BEAT_COOLDOWN or bt == last_beat:
                z += get_beat_pulse(tsb, BEAT_PULSE_DURATION, height)
                if tsb < (1/hz):
                    last_beat = bt
            break
    return z, last_beat


def emergency_stop(crazyflies):
    print("\nEMERGENCY STOP")
    for cf in crazyflies:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.04, duration=2.0)


# =========================================================
# DRONE 1 CHOREOGRAPHY
# =========================================================

def run_drone1(cf, features):
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times'][features['beat_times'] <= SONG_END]
    drop_times = features['drop_times'][features['drop_times'] <= SONG_END]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.7
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    initPos   = cf.position()
    timesteps = np.arange(0, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    c1_angle  = 0.0
    c2_angle  = 0.0
    s2_angle  = 0.0
    orb_angle = 0.0
    fin_angle = 0.0
    prev_x = D1_START[0]
    prev_y = D1_START[1]

    for t in timesteps:
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

            # 0:00-0:14  BPM hops drifting toward fabric center
            if t < CIRCLE1_START:
                dx, dy = fabric_drift(t, drone_id=1, drift_speed=0.07)
                x = FABRIC_CENTER_X + dx
                y = FABRIC_CENTER_Y + dy
                hop = BEAT_PULSE_HEIGHT * (0.4 + vocal * 0.6)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, hop, Hz)

            # 0:15-0:24  Circle with lyrics
            elif t <= CIRCLE1_END:
                c1_angle += speed * SPEED_SCALE * 0.07
                radius = 0.6 + vocal * 0.5
                x = D1_START[0] + radius * np.cos(c1_angle)
                y = D1_START[1] + radius * np.sin(c1_angle)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.5, Hz)

            # 0:24-0:40  BPM + spirals drifting toward fabric
            elif t < CIRCLE2_START:
                dx, dy = fabric_drift(t, drone_id=1, drift_speed=0.06)
                x = FABRIC_CENTER_X + dx
                y = FABRIC_CENTER_Y + dy
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, Hz)
                if ENABLE_SPIRAL:
                    for dt in drop_times:
                        if abs(t - dt) < (1/Hz):
                            if (dt - last_spiral) >= SPIRAL_COOLDOWN:
                                spiral_start = t
                                last_spiral  = t
                            break
                    tss = t - spiral_start
                    if 0 <= tss <= SPIRAL_DURATION:
                        sx, sy, sz = get_spiral_offset(tss)
                        x += sx; y += sy; z += sz

            # 0:40-0:50  Second circle, lower
            elif t <= CIRCLE2_END:
                c2_angle += speed * SPEED_SCALE * 0.06
                radius = 0.5 + vocal * 0.4
                x = D1_START[0] + radius * np.cos(c2_angle)
                y = D1_START[1] + radius * np.sin(c2_angle)
                z = 0.8 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.6, Hz)

            # 0:50-0:55  Tornado rising
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

            # 0:55-1:04  Wide lissajous
            if t < BOTH_CIRCLE_START:
                s2_angle += speed * SPEED_SCALE * 0.045
                x = D1_START[0] + 1.1 * np.sin(s2_angle)
                y = D1_START[1] + 1.0 * np.sin(2 * s2_angle + np.pi/4)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, Hz)

            # 1:04-1:19  Both circle — drone 1 faster at 0 degrees
            elif t <= BOTH_CIRCLE_END:
                orb_angle += 1.1 * SPEED_SCALE * (1/Hz)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                x = 1.0 * np.cos(orb_angle) * fade
                y = 1.0 * np.sin(orb_angle) * fade
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.7, Hz)

            # 1:19-1:30  Sway left/right
            elif t <= SWAY_END:
                fade = smooth_fade(t, SWAY_START, duration=1.5)
                beat_idx = int(np.searchsorted(beat_times, t)) % 4
                sway_dir = 1.0 if beat_idx < 2 else -1.0
                x = sway_dir * 1.0 * vocal * fade
                y = D1_START[1] * (1 - fade * 0.5)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 1.1, Hz)

            # 1:30-1:51  Wide orbit + strong hops while drone 2 spirals
            else:
                orb_angle += 0.7 * SPEED_SCALE * (1/Hz)
                x = 1.0 * np.cos(orb_angle)
                y = 0.8 * np.sin(orb_angle)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 1.4, Hz)

        # ===== SECTION 3 =====
        else:
            z_min, z_max = D1_Z_MIN, D1_Z_MAX
            fade = smooth_fade(t, S2_END, duration=BLEND_DURATION)

            # 1:51-2:11  Up/down alternating + elliptical orbit over fabric
            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 8.0
                ox = FABRIC_X_HALF * np.cos(orbit_s3 * (t - S2_END))
                oy = FABRIC_Y_HALF * np.sin(orbit_s3 * (t - S2_END))
                x = FABRIC_CENTER_X + ox * fade
                y = FABRIC_CENTER_Y + oy * fade
                z = 2.0 + th + updown_alt(t, drone_id=1, amplitude=0.35) * fade

            # 2:11-end  Final circle expands then contracts
            else:
                fin_angle += 2 * np.pi / 9.0 * (1/Hz)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                if progress < 0.5:
                    radius = 0.4 + progress * 1.2
                else:
                    radius = 1.0 * (1 - (progress - 0.5) / 0.5)
                x = radius * np.cos(fin_angle)
                y = radius * np.sin(fin_angle)
                z = 1.8 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, Hz)

        # Boundary smoothing
        if abs(t - S1_END) < 2.0 and t > S1_END:
            bf = smooth_fade(t, S1_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)
        if abs(t - S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)

        # Landing — blend to 0,0 so initPos returns drone home
        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, 0.0, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        r, g, b = get_color(t, beat_times)
        send_color(cf, r, g, b)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 1 complete.")


# =========================================================
# DRONE 2 CHOREOGRAPHY
# =========================================================

def run_drone2(cf, features):
    rms_times      = features['rms_times']
    rms            = features['rms']
    bass           = features['bass_energy']
    treble         = features['treble_energy']
    freq_times     = features['freq_times']
    beat_times_full = features['beat_times'][features['beat_times'] <= SONG_END]
    beat_times     = beat_times_full[beat_times_full >= S1_END]
    drop_times     = features['drop_times'][features['drop_times'] <= SONG_END]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.7
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    initPos   = cf.position()
    timesteps = np.arange(S1_END, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    orb_angle = np.pi   # starts opposite to drone 1
    fin_angle = 0.0
    prev_x = D2_START[0]
    prev_y = D2_START[1]

    for t in timesteps:
        speed = get_value(t, rms_times, rms_norm)
        size  = get_value(t, freq_times, bass_norm)
        th    = get_value(t, freq_times, treb_norm)
        vocal = get_value(t, rms_times, rms_long_norm)

        x = D2_START[0]
        y = D2_START[1]
        z = 0.9 + th
        z_min, z_max = D2_Z_MIN, D2_Z_MAX
        fade_in = smooth_fade(t, S1_END, duration=BLEND_DURATION)

        # ===== SECTION 2 =====
        if t <= S2_END:

            # 0:55-1:04  Violin bow arc — full ellipse
            if t < BOTH_CIRCLE_START:
                ts = t - ARC_START
                bx = 0.9 * np.cos(1.8 * SPEED_SCALE * ts)
                by = 1.1 * np.sin(1.8 * SPEED_SCALE * ts)
                x = D2_START[0] + bx * fade_in
                y = D2_START[1] + by * fade_in
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.7, Hz)

            # 1:04-1:19  Circle at 180 degrees, slower than drone 1
            elif t <= BOTH_CIRCLE_END:
                orb_angle += 0.7 * SPEED_SCALE * (1/Hz)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                x = 1.0 * np.cos(orb_angle + np.pi) * fade
                y = 1.0 * np.sin(orb_angle + np.pi) * fade
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.7, Hz)

            # 1:19-1:30  Sway together with drone 1
            elif t <= SWAY_END:
                fade = smooth_fade(t, SWAY_START, duration=1.5)
                beat_idx = int(np.searchsorted(beat_times, t)) % 4
                sway_dir = 1.0 if beat_idx < 2 else -1.0
                x = sway_dir * 1.0 * vocal * fade
                y = D2_START[1] * (1 - fade * 0.5)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT * 0.9, Hz)

            # 1:30-1:51  Big ascending helix with smooth descent at end
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

            # 1:51-2:11  Up/down alternating + orbit over fabric
            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 8.0
                base_angle = 2 * np.pi / 3
                ox = FABRIC_X_HALF * np.cos(orbit_s3*(t-S2_END) + base_angle)
                oy = FABRIC_Y_HALF * np.sin(orbit_s3*(t-S2_END) + base_angle)
                dx, dy = fabric_drift(t, drone_id=2, drift_speed=0.04)
                x = FABRIC_CENTER_X + ox*fade + dx*0.2*fade
                y = FABRIC_CENTER_Y + oy*fade + dy*0.2*fade
                z = 1.3 + th + updown_alt(t, drone_id=2, amplitude=0.35) * fade

            # 2:11-end  Middle ring, 1.5s chase delay
            else:
                fin_angle += 2 * np.pi / 9.0 * (1/Hz)
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
                                                z, BEAT_PULSE_HEIGHT * 0.8, Hz)

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

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        r, g, b = get_color(t, beat_times_full)
        send_color(cf, r, g, b)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 2 complete.")


# =========================================================
# DRONE 3 CHOREOGRAPHY
# =========================================================

def run_drone3(cf, features):
    rms_times      = features['rms_times']
    rms            = features['rms']
    treble         = features['treble_energy']
    beat_times_full = features['beat_times'][features['beat_times'] <= SONG_END]
    beat_times     = beat_times_full[beat_times_full >= S2_END]
    drop_times     = features['drop_times']
    drop_times     = drop_times[(drop_times >= S2_END) & (drop_times <= SONG_END)]

    rms_s  = smooth(rms,    window=60)
    treb_s = smooth(treble, window=60)

    rms_norm  = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.7
    treb_norm = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.25

    initPos   = cf.position()
    timesteps = np.arange(S2_END, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    fin_angle = 0.0

    for t in timesteps:
        th      = get_value(t, rms_times, treb_norm)
        fade_in = smooth_fade(t, S2_END + 1.0, duration=BLEND_DURATION)

        x = D3_START[0]
        y = D3_START[1]
        z = 0.8 + th
        z_min, z_max = D3_Z_MIN, D3_Z_MAX

        # 1:51-2:11  Up/down opposite + orbit over fabric
        if t <= UPDOWN_END:
            orbit_s3 = 2 * np.pi / 8.0
            base_angle = 4 * np.pi / 3
            ox = FABRIC_X_HALF * np.cos(orbit_s3*(t-S2_END) + base_angle)
            oy = FABRIC_Y_HALF * np.sin(orbit_s3*(t-S2_END) + base_angle)
            dx, dy = fabric_drift(t, drone_id=3, drift_speed=0.04)
            x = FABRIC_CENTER_X + ox*fade_in + dx*0.2*fade_in
            y = FABRIC_CENTER_Y + oy*fade_in + dy*0.2*fade_in
            z = 0.8 + th + updown_alt(t, drone_id=3, amplitude=0.35) * fade_in

        # 2:11-end  Innermost ring, 3s chase delay
        else:
            fin_angle += 2 * np.pi / 9.0 * (1/Hz)
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
                                            z, BEAT_PULSE_HEIGHT * 0.7, Hz)

            if ENABLE_SPIRAL:
                for dt in drop_times:
                    if abs(t - dt) < (1/Hz):
                        if (dt - last_spiral) >= SPIRAL_COOLDOWN:
                            spiral_start = t
                            last_spiral  = t
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

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        r, g, b = get_color(t, beat_times_full)
        send_color(cf, r, g, b)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 3 complete.")


# =========================================================
# MAIN
# =========================================================

def main():
    global SIM, DRY_RUN

    if DRY_RUN:
        print("DRY RUN")
        features = extract_features(AUDIO_PATH)
        print(f"Duration: {features['duration']:.1f}s")
        print(f"Beats:    {len(features['beat_times'])}")
        print(f"\nSchedule:")
        print(f"  t=0s       D1 takeoff + choreography")
        print(f"  t={DRONE2_TAKEOFF_DELAY}s    D2 takeoff")
        print(f"  t={S1_END}s    D2 choreography starts")
        print(f"  t={DRONE3_TAKEOFF_DELAY}s   D3 takeoff")
        print(f"  t={S2_END}s   D3 choreography starts")
        print(f"  t={SONG_END}s D3 all land")
        print(f"\nStart positions:")
        print(f"  D1: {D1_START}")
        print(f"  D2: {D2_START}")
        print(f"  D3: {D3_START}")
        return

    if SIM:
        print("SIMULATION mode")
        from pycrazyswarm import Crazyswarm
        swarm = Crazyswarm(args='--vis=null --sim')
    else:
        print("REAL DRONES")
        from crazyflie_py import Crazyswarm
        swarm = Crazyswarm()

    crazyflies = swarm.allcfs.crazyflies
    timeHelper = swarm.timeHelper
    print(f"Detected {len(crazyflies)} drone(s)")

    cf1 = crazyflies[0] if len(crazyflies) > 0 else None
    cf2 = crazyflies[1] if len(crazyflies) > 1 else None
    cf3 = crazyflies[2] if len(crazyflies) > 2 else None

    print("Extracting audio features...")
    features = extract_features(AUDIO_PATH)
    print(f"Ready. Duration: {features['duration']:.1f}s")

    # Drone 1 takes off first
    print("Drone 1 taking off...")
    cf1.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    time.sleep(TAKEOFF_DURATION + 1.0)

    perf_start = time.time()

    def drone1_thread():
        try:
            run_drone1(cf1, features)
        except Exception as e:
            print(f"D1 error: {e}")

    def drone2_thread():
        elapsed = time.time() - perf_start
        remaining = DRONE2_TAKEOFF_DELAY - elapsed
        if remaining > 0:
            time.sleep(remaining)
        if cf2 is None:
            print("No drone 2")
            return
        print("Drone 2 taking off...")
        cf2.takeoff(targetHeight=DRONE2_TAKEOFF_HEIGHT,
                    duration=DRONE2_TAKEOFF_DURATION)
        time.sleep(DRONE2_TAKEOFF_DURATION + 1.5)
        elapsed = time.time() - perf_start
        remaining = S1_END - elapsed
        if remaining > 0:
            time.sleep(remaining)
        print("Drone 2 choreography starting...")
        try:
            run_drone2(cf2, features)
        except Exception as e:
            print(f"D2 error: {e}")

    def drone3_thread():
        elapsed = time.time() - perf_start
        remaining = DRONE3_TAKEOFF_DELAY - elapsed
        if remaining > 0:
            time.sleep(remaining)
        if cf3 is None:
            print("No drone 3")
            return
        print("Drone 3 taking off...")
        cf3.takeoff(targetHeight=DRONE3_TAKEOFF_HEIGHT,
                    duration=DRONE3_TAKEOFF_DURATION)
        time.sleep(DRONE3_TAKEOFF_DURATION + 1.5)
        elapsed = time.time() - perf_start
        remaining = S2_END - elapsed
        if remaining > 0:
            time.sleep(remaining)
        print("Drone 3 choreography starting...")
        try:
            run_drone3(cf3, features)
        except Exception as e:
            print(f"D3 error: {e}")

    t1 = threading.Thread(target=drone1_thread)
    t2 = threading.Thread(target=drone2_thread)
    t3 = threading.Thread(target=drone3_thread)

    try:
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
    except KeyboardInterrupt:
        emergency_stop(crazyflies)
        return

    print("Landing all drones...")
    for cf in [cf1, cf2, cf3]:
        if cf is not None:
            cf.land(targetHeight=0.04, duration=LAND_DURATION)
    time.sleep(LAND_DURATION + 1.0)
    print("Landed successfully.")


if __name__ == '__main__':
    main()