import numpy as np
import time
from types import SimpleNamespace
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
SPEED_SCALE = 0.75

# ---- SAFETY BOUNDS — tightened after net collision ----
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
Z_MIN, Z_MAX = 0.5, 2.5

# ---- FIXED START POSITIONS ----
D1_START = np.array([0.0,   0.5,  0.0])
D2_START = np.array([-0.5, -0.5,  0.0])
D3_START = np.array([0.5,  -0.5,  0.0])

# ---- HEIGHT ZONES ----
D1_Z_MIN, D1_Z_MAX = 0.5, 2.5
D2_Z_MIN, D2_Z_MAX = 0.5, 1.8
D3_Z_MIN, D3_Z_MAX = 0.5, 2.0

# ---- BEAT SETTINGS ----
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.40
BEAT_PULSE_DURATION = 0.8

# ---- SPIRAL SETTINGS ----
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 6.0
SPIRAL_HEIGHT = 0.8
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 1.2   # reduced from 1.5 to stay within tighter bounds
ENABLE_SPIRAL = True
BLEND_DURATION = 3.0

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
D2_SPIRAL_START   = 90.0
D2_SPIRAL_END     = 111.0
UPDOWN_START      = 116.0
UPDOWN_END        = 131.0
FINAL_CIRCLE_START = 131.0

# ---- DRONE 2/3 TAKEOFF TIMING ----
DRONE2_TAKEOFF_DELAY   = 50.0
DRONE2_TAKEOFF_HEIGHT  = 0.8
DRONE2_TAKEOFF_DURATION = 3.0
DRONE3_TAKEOFF_DELAY   = 106.0
DRONE3_TAKEOFF_HEIGHT  = 0.8
DRONE3_TAKEOFF_DURATION = 3.0

def clamp_position(position, z_min=Z_MIN, z_max=Z_MAX):
    x = np.clip(position[0], X_MIN, X_MAX)
    y = np.clip(position[1], Y_MIN, Y_MAX)
    z = np.clip(position[2], z_min, z_max)
    return np.array([x, y, z])

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

def tornado(t, start_time, duration=5.0, max_radius=1.2, max_height=1.5):
    """Slow upward tornado — starts tiny at bottom, grows wider as it rises."""
    t_since = t - start_time
    if t_since < 0 or t_since > duration:
        return 0.0, 0.0, 0.0
    p = t_since / duration
    angle = p * 4 * np.pi * SPEED_SCALE
    radius = max_radius * p
    return radius * np.cos(angle), radius * np.sin(angle), max_height * p

def helix_spiral(t, start_time, duration=21.0, max_radius=1.0, max_height=1.0):
    """Ascending helix for drone 2."""
    t_since = t - start_time
    if t_since < 0 or t_since > duration:
        return 0.0, 0.0, 0.0
    p = t_since / duration
    angle = p * 6 * np.pi * SPEED_SCALE
    return max_radius * p * np.cos(angle), max_radius * p * np.sin(angle), max_height * p

def updown_alternating(t, drone_id, amplitude=0.4):
    phase = 0.0 if drone_id in [1, 2] else np.pi
    return amplitude * np.sin(2 * np.pi * t / 3.2 + phase)

def smooth_fade(t, start, duration=2.0):
    t_since = t - start
    if t_since <= 0: return 0.0
    if t_since >= duration: return 1.0
    return np.sin(np.pi * t_since / (2 * duration)) ** 2

def blend(a, b, factor):
    return a * (1 - factor) + b * factor

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

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.6
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    initPos = cf.position()
    timesteps = np.arange(0, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    circle1_angle = 0.0
    circle2_angle = 0.0
    s2_angle      = 0.0
    orbit_angle   = 0.0
    final_angle   = 0.0
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

        # =================== SECTION 1 ===================
        if t <= S1_END:
            if t < CIRCLE1_START:
                x = D1_START[0]
                y = D1_START[1]
                hop = BEAT_PULSE_HEIGHT * (0.3 + vocal * 0.7)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z, hop, Hz)

            elif t <= CIRCLE1_END:
                circle1_angle += speed * SPEED_SCALE * 0.06
                radius = 0.6 + vocal * 0.5   # reduced radius
                x = D1_START[0] + radius * np.cos(circle1_angle)
                y = D1_START[1] + radius * np.sin(circle1_angle)

            elif t < CIRCLE2_START:
                x = D1_START[0]
                y = D1_START[1]
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, Hz)
                if ENABLE_SPIRAL:
                    for dt in drop_times:
                        if abs(t - dt) < (1/Hz):
                            if (dt - last_spiral) >= SPIRAL_COOLDOWN:
                                spiral_start = t
                                last_spiral = t
                            break
                    tss = t - spiral_start
                    if 0 <= tss <= SPIRAL_DURATION:
                        sx, sy, sz = get_spiral_offset(tss)
                        x += sx; y += sy; z += sz

            elif t <= CIRCLE2_END:
                circle2_angle += speed * SPEED_SCALE * 0.05
                radius = 0.5 + vocal * 0.4   # reduced radius
                x = D1_START[0] + radius * np.cos(circle2_angle)
                y = D1_START[1] + radius * np.sin(circle2_angle)
                z = 0.7 + th

            elif t <= TORNADO_END:
                tx, ty, tz = tornado(t, TORNADO_START,
                                     duration=TORNADO_END - TORNADO_START,
                                     max_radius=1.2, max_height=1.5)
                fade = smooth_fade(t, TORNADO_START, duration=1.0)
                x = D1_START[0] + tx * fade
                y = D1_START[1] + ty * fade
                z = 0.5 + tz

            else:
                x = D1_START[0]
                y = D1_START[1]

        # =================== SECTION 2 ===================
        elif t <= S2_END:
            z = 1.8 + th
            z_min, z_max = 1.5, 2.5

            if t < BOTH_CIRCLE_START:
                s2_angle += speed * SPEED_SCALE * 0.04
                # Reduced lissajous amplitude to stay in bounds
                x = D1_START[0] + 1.2 * np.sin(s2_angle)
                y = D1_START[1] + 1.2 * np.sin(2 * s2_angle + np.pi / 4)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, Hz)

            elif t <= BOTH_CIRCLE_END:
                orbit_angle += 1.0 * SPEED_SCALE * (1/Hz)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                # Reduced circle radius
                x = D1_START[0] + 1.0 * np.cos(orbit_angle) * fade
                y = D1_START[1] + 1.0 * np.sin(orbit_angle) * fade

            elif t <= SWAY_END:
                beat_idx = int(np.searchsorted(beat_times, t)) % 8
                sway_dir = 1.0 if beat_idx < 4 else -1.0
                fade = smooth_fade(t, SWAY_START, duration=2.0)
                x = D1_START[0] + sway_dir * 1.0 * vocal * fade
                y = D1_START[1]
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, Hz)

            else:
                # Slow wide orbit while drone 2 spirals
                orbit_angle += 0.6 * SPEED_SCALE * (1/Hz)
                x = D1_START[0] + 1.1 * np.cos(orbit_angle)
                y = D1_START[1] + 1.1 * np.sin(orbit_angle)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 1.3, Hz)

        # =================== SECTION 3 ===================
        else:
            z_min, z_max = D1_Z_MIN, D1_Z_MAX
            fade = smooth_fade(t, S2_END, duration=BLEND_DURATION)

            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 10.0
                ox = 0.5 * np.cos(orbit_s3 * (t - S2_END))
                oy = 0.5 * np.sin(orbit_s3 * (t - S2_END))
                x = D1_START[0] + ox * fade
                y = D1_START[1] + oy * fade
                z = 2.0 + th + updown_alternating(t, drone_id=1, amplitude=0.3) * fade

            else:
                final_angle += 2 * np.pi / 10.0 * (1/Hz)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                radius = 1.0 if progress < 0.7 else 1.0 * (1 - (progress - 0.7) / 0.3)
                x = radius * np.cos(final_angle)   # centered on origin
                y = radius * np.sin(final_angle)
                z = 2.0 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT, Hz)

        # Smooth transition at section boundaries
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
            x = blend(x, 0.0, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 1 choreography complete.")

def run_drone2(cf, features):
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= S1_END) & (beat_times <= SONG_END)]
    drop_times = features['drop_times'][features['drop_times'] <= SONG_END]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.6
    bass_norm     = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm     = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3
    rms_long_norm = (rms_long - rms_long.min()) / (rms_long.max() - rms_long.min() + 1e-8)

    initPos = cf.position()
    timesteps = np.arange(S1_END, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    orbit_angle  = np.pi
    final_angle  = 0.0
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

        # =================== SECTION 2 ===================
        if t <= S2_END:

            if t < BOTH_CIRCLE_START:
                # Elliptical bow arc
                t_since = t - ARC_START
                bx = 0.8 * np.cos(1.5 * SPEED_SCALE * t_since)
                by = 1.1 * np.sin(1.5 * SPEED_SCALE * t_since)
                x = D2_START[0] + bx * fade_in
                y = D2_START[1] + by * fade_in

            elif t <= BOTH_CIRCLE_END:
                orbit_angle += 0.6 * SPEED_SCALE * (1/Hz)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                x = D2_START[0] + 1.0 * np.cos(orbit_angle) * fade
                y = D2_START[1] + 1.0 * np.sin(orbit_angle) * fade

            elif t <= SWAY_END:
                beat_idx = int(np.searchsorted(beat_times, t)) % 8
                sway_dir = 1.0 if beat_idx < 4 else -1.0
                fade = smooth_fade(t, SWAY_START, duration=2.0)
                x = D2_START[0] + sway_dir * 1.0 * vocal * fade
                y = D2_START[1]
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 0.8, Hz)

            else:
                # Ascending helix with descent at end
                helix_dur = D2_SPIRAL_END - D2_SPIRAL_START
                descent_start = D2_SPIRAL_END - 5.0

                if t < descent_start:
                    hx, hy, hz = helix_spiral(t, D2_SPIRAL_START,
                                              duration=helix_dur,
                                              max_radius=1.0, max_height=1.0)
                    x = D2_START[0] + hx
                    y = D2_START[1] + hy
                    z = 0.8 + hz + th
                    z_min, z_max = D2_Z_MIN, 2.2
                else:
                    dp = (t - descent_start) / 5.0
                    hx, hy, hz = helix_spiral(descent_start, D2_SPIRAL_START,
                                              duration=helix_dur,
                                              max_radius=1.0, max_height=1.0)
                    peak_x = D2_START[0] + hx
                    peak_y = D2_START[1] + hy
                    peak_z = 0.8 + hz + th
                    x = blend(peak_x, D2_START[0], dp)
                    y = blend(peak_y, D2_START[1], dp)
                    z = blend(peak_z, 0.8, dp)
                    z_min, z_max = D2_Z_MIN, 2.2

        # =================== SECTION 3 ===================
        else:
            z_min, z_max = Z_MIN, Z_MAX
            fade = smooth_fade(t, S2_END + 2.0, duration=BLEND_DURATION)

            if t <= UPDOWN_END:
                orbit_s3 = 2 * np.pi / 10.0
                base_angle = 2 * np.pi / 3
                ox = 0.5 * np.cos(orbit_s3 * (t - S2_END) + base_angle)
                oy = 0.5 * np.sin(orbit_s3 * (t - S2_END) + base_angle)
                x = D2_START[0] + ox * fade
                y = D2_START[1] + oy * fade
                z = 1.3 + th + updown_alternating(t, drone_id=2, amplitude=0.3) * fade

            else:
                final_angle += 2 * np.pi / 10.0 * (1/Hz)
                progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
                radius = 0.7 if progress < 0.7 else 0.7 * (1 - (progress - 0.7) / 0.3)
                delay_angle = final_angle - (1.5 * 2 * np.pi / 10.0)
                x = radius * np.cos(delay_angle)
                y = radius * np.sin(delay_angle)
                z = 1.3 + th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                                BEAT_PULSE_HEIGHT * 0.8, Hz)

        # Smooth section 3 transition
        if abs(t - S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf)
            y = blend(prev_y, y, bf)

        # Landing
        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, -0.3, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 2 choreography complete.")

def run_drone3(cf, features):
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= S2_END) & (beat_times <= SONG_END)]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= S2_END) & (drop_times <= SONG_END)]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)

    rms_norm  = 0.7 + ((rms_s - rms_s.min()) / (rms_s.max() - rms_s.min() + 1e-8)) * 0.6
    bass_norm = 0.5 + ((bass_s - bass_s.min()) / (bass_s.max() - bass_s.min() + 1e-8)) * 1.0
    treb_norm = 0.0 + ((treb_s - treb_s.min()) / (treb_s.max() - treb_s.min() + 1e-8)) * 0.3

    initPos = cf.position()
    timesteps = np.arange(S2_END, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    final_angle  = 0.0
    prev_x = D3_START[0]
    prev_y = D3_START[1]

    for t in timesteps:
        speed = get_value(t, rms_times, rms_norm)
        size  = get_value(t, freq_times, bass_norm)
        th    = get_value(t, freq_times, treb_norm)

        fade_in = smooth_fade(t, S2_END + 1.0, duration=BLEND_DURATION)

        x = D3_START[0]
        y = D3_START[1]
        z = 0.8 + th
        z_min, z_max = D3_Z_MIN, D3_Z_MAX

        if t <= UPDOWN_END:
            orbit_s3 = 2 * np.pi / 10.0
            base_angle = 4 * np.pi / 3
            ox = 0.5 * np.cos(orbit_s3 * (t - S2_END) + base_angle)
            oy = 0.5 * np.sin(orbit_s3 * (t - S2_END) + base_angle)
            x = D3_START[0] + ox * fade_in
            y = D3_START[1] + oy * fade_in
            z = 0.8 + th + updown_alternating(t, drone_id=3, amplitude=0.3) * fade_in

        else:
            final_angle += 2 * np.pi / 10.0 * (1/Hz)
            progress = (t - FINAL_CIRCLE_START) / (SONG_END - FINAL_CIRCLE_START)
            radius = 0.4 if progress < 0.7 else 0.4 * (1 - (progress - 0.7) / 0.3)
            delay_angle = final_angle - (3.0 * 2 * np.pi / 10.0)
            x = radius * np.cos(delay_angle) * fade_in
            y = radius * np.sin(delay_angle) * fade_in
            z = 0.8 + th
            z_min, z_max = Z_MIN, Z_MAX

            z, last_beat = apply_beat_pulse(t, beat_times, last_beat, z,
                                            BEAT_PULSE_HEIGHT * 0.7, Hz)

            if ENABLE_SPIRAL:
                for dt in drop_times:
                    if abs(t - dt) < (1/Hz):
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

        # Landing
        if t >= land_start:
            lp = min((t - land_start) / 4.0, 1.0)
            x = blend(x, 0.3, lp)
            y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp)
            z_min, z_max = Z_MIN, Z_MAX

        prev_x = x
        prev_y = y

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 3 choreography complete.")

def main():
    global SIM, DRY_RUN

    if DRY_RUN:
        print("DRY RUN - validating choreography")
        features = extract_features(AUDIO_PATH)
        print(f"Audio duration: {features['duration']:.1f}s")
        print(f"Beats detected: {len(features['beat_times'])}")
        print(f"Drop times: {len(features['drop_times'])}")
        print(f"\nSection splits:")
        for i, t in enumerate(features['section_times']):
            print(f"  Section {i}: {t:.1f}s")
        print(f"\nChoreography schedule:")
        print(f"  t=0s      Drone 1 takeoff")
        print(f"  t=4s      Drone 1 choreography starts")
        print(f"  t={DRONE2_TAKEOFF_DELAY}s   Drone 2 takeoff")
        print(f"  t={S1_END}s    Drone 2 choreography starts")
        print(f"  t={DRONE3_TAKEOFF_DELAY}s  Drone 3 takeoff")
        print(f"  t={S2_END}s   Drone 3 choreography starts")
        print(f"  t={SONG_END}s  All drones land")
        print("\nDry run complete.")
        return

    if SIM:
        print("Running in SIMULATION mode")
        from pycrazyswarm import Crazyswarm
        swarm = Crazyswarm(args='--vis=null --sim')
    else:
        print("Running on REAL DRONES")
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
    print(f"Audio ready. Duration: {features['duration']:.1f}s")

    # ---- DRONE 1 TAKEOFF ----
    print(f"Drone 1 taking off to {TAKEOFF_HEIGHT}m...")
    cf1.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    time.sleep(TAKEOFF_DURATION + 1.0)

    # Track performance start time
    perf_start = time.time()

    def drone1_thread():
        try:
            run_drone1(cf1, features)
        except Exception as e:
            print(f"Drone 1 error: {e}")

    def drone2_thread():
        # Wait for takeoff delay
        elapsed = time.time() - perf_start
        remaining = DRONE2_TAKEOFF_DELAY - elapsed
        if remaining > 0:
            time.sleep(remaining)

        if cf2 is None:
            print("No drone 2 available")
            return

        print(f"Drone 2 taking off to {DRONE2_TAKEOFF_HEIGHT}m...")
        cf2.takeoff(targetHeight=DRONE2_TAKEOFF_HEIGHT,
                    duration=DRONE2_TAKEOFF_DURATION)
        time.sleep(DRONE2_TAKEOFF_DURATION + 1.5)

        # Wait until S1_END before starting choreography
        elapsed = time.time() - perf_start
        remaining = S1_END - elapsed
        if remaining > 0:
            time.sleep(remaining)

        print("Drone 2 choreography starting...")
        try:
            run_drone2(cf2, features)
        except Exception as e:
            print(f"Drone 2 error: {e}")

    def drone3_thread():
        # Wait for takeoff delay
        elapsed = time.time() - perf_start
        remaining = DRONE3_TAKEOFF_DELAY - elapsed
        if remaining > 0:
            time.sleep(remaining)

        if cf3 is None:
            print("No drone 3 available")
            return

        print(f"Drone 3 taking off to {DRONE3_TAKEOFF_HEIGHT}m...")
        cf3.takeoff(targetHeight=DRONE3_TAKEOFF_HEIGHT,
                    duration=DRONE3_TAKEOFF_DURATION)
        time.sleep(DRONE3_TAKEOFF_DURATION + 1.5)

        # Wait until S2_END before starting choreography
        elapsed = time.time() - perf_start
        remaining = S2_END - elapsed
        if remaining > 0:
            time.sleep(remaining)

        print("Drone 3 choreography starting...")
        try:
            run_drone3(cf3, features)
        except Exception as e:
            print(f"Drone 3 error: {e}")

    # Start all threads
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

    # ---- LAND ALL ----
    print("Landing all drones...")
    for cf in crazyflies:
        if cf is not None:
            cf.land(targetHeight=0.04, duration=LAND_DURATION)
    time.sleep(LAND_DURATION + 1.0)
    print("Landed successfully.")

if __name__ == '__main__':
    main()