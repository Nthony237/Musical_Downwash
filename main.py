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

# ---- LED COLOR — set once at takeoff, never changed during flight ----
# ---- LED COLORS — Powerpuff Girls, one per drone ----
D1_LED = (255,  20,  80)   
D2_LED = ( 20, 120, 255)   
D3_LED = (  0, 200,  50)   

# ---- SAFETY BOUNDS ----
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
Z_MIN, Z_MAX = 0.5, 2.5

# ---- START POSITIONS ----
D1_START = np.array([0.0,   1.0,  0.0])
D2_START = np.array([-1.0, -1.0,  0.0])
D3_START = np.array([1.0,  -1.0,  0.0])

# ---- FABRIC STRUCTURE ----
FABRIC_CENTER_X = 0.0
FABRIC_CENTER_Y = 0.0
FABRIC_X_HALF   = 0.5
FABRIC_Y_HALF   = 1.0
FABRIC_HEIGHT   = 0.46
DOWNWASH_HEIGHT = 0.82
DOWNWASH_HIGH   = 1.6

# ---- HEIGHT ZONES ----
D1_Z_MIN, D1_Z_MAX = 0.5, 2.5
D2_Z_MIN, D2_Z_MAX = 0.5, 1.6
D3_Z_MIN, D3_Z_MAX = 0.5, 1.9

# ---- BEAT SETTINGS ----
BEAT_COOLDOWN       = 1.6
BEAT_PULSE_HEIGHT   = 0.45
BEAT_PULSE_DURATION = 0.8

# ---- SPIRAL SETTINGS ----
SPIRAL_DURATION     = 4.0
SPIRAL_COOLDOWN     = 6.0
SPIRAL_HEIGHT       = 0.8
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END   = 1.1
ENABLE_SPIRAL       = True
BLEND_DURATION      = 3.0

# ---- SECTION TIMESTAMPS ----
S1_END   = 55.0
S2_END   = 111.0

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
D2_HELIX_START     = 90.0
D2_HELIX_END       = 111.0
UPDOWN_START       = 111.0
UPDOWN_END         = 131.0
FINAL_CIRCLE_START = 131.0

# ---- BROKEN HEART ----
HEART_START = 143.0
HEART_BREAK = 158.0
HEART_END   = SONG_END

# ---- DRONE 2/3 TAKEOFF TIMING ----
DRONE2_TAKEOFF_DELAY    = 50.0
DRONE2_TAKEOFF_HEIGHT   = 0.8
DRONE2_TAKEOFF_DURATION = 3.0
DRONE3_TAKEOFF_DELAY    = 106.0
DRONE3_TAKEOFF_HEIGHT   = 0.8
DRONE3_TAKEOFF_DURATION = 3.0


# =========================================================
# HELPERS
# =========================================================

def set_led(cf, color):
    try:
        cf.setParam('ring.effect',     7)
        cf.setParam('ring.solidRed',   color[0])
        cf.setParam('ring.solidGreen', color[1])
        cf.setParam('ring.solidBlue',  color[2])
    except Exception:
        pass


def heart_point(param):
    hx = 16 * np.sin(param) ** 3
    hy = (13*np.cos(param) - 5*np.cos(2*param)
          - 2*np.cos(3*param) - np.cos(4*param))
    hx /= 16.0
    hy /= 13.0
    lab_x = hy * 0.5
    lab_y = hx * 0.6
    return lab_x, lab_y


def get_heart_half(t, side, start_time, duration=8.0, break_offset=0.0):
    progress = np.clip((t - start_time) / duration, 0.0, 1.0)
    if side == 'left':
        param = np.pi + progress * np.pi
        y_sign = -1.0
    else:
        param = progress * np.pi
        y_sign = 1.0
    x, y = heart_point(param)
    return x, y + y_sign * break_offset


def clamp_position(position, z_min=Z_MIN, z_max=Z_MAX):
    return np.array([
        np.clip(position[0], X_MIN, X_MAX),
        np.clip(position[1], Y_MIN, Y_MAX),
        np.clip(position[2], z_min, z_max)
    ])


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
    return r*np.sin(angle), r*np.cos(angle), SPIRAL_HEIGHT * p


def rotated_circle(angle, radius):
    return radius * np.sin(angle), radius * np.cos(angle)


def tornado_move(t, start, duration=5.0, max_r=1.0, max_h=1.5):
    ts = t - start
    if ts < 0 or ts > duration:
        return 0.0, 0.0, 0.0
    p = ts / duration
    angle = p * 4 * np.pi * SPEED_SCALE
    return max_r*p*np.sin(angle), max_r*p*np.cos(angle), max_h*p


def helix_move(t, start, duration=21.0, max_r=1.0, max_h=1.0):
    ts = t - start
    if ts < 0 or ts > duration:
        return 0.0, 0.0, 0.0
    p = ts / duration
    angle = p * 6 * np.pi * SPEED_SCALE
    return max_r*p*np.sin(angle), max_r*p*np.cos(angle), max_h*p


def updown_alt(t, drone_id, amplitude=0.35):
    phase = 0.0 if drone_id in [1, 2] else np.pi
    return amplitude * np.sin(2 * np.pi * t / 3.2 + phase)


def fabric_sweep(t, drone_id, sweep_speed=0.10):
    phases = {1: 0.0, 2: 2*np.pi/3, 3: 4*np.pi/3}
    phase = phases[drone_id]
    x_offset = 0.9 * np.sin(sweep_speed * SPEED_SCALE * t + phase)
    y_offset = 1.3 * np.sin(sweep_speed * SPEED_SCALE * t + phase + np.pi/4)
    cycle = (np.sin(sweep_speed * SPEED_SCALE * t + phase) + 1) / 2
    z_height = DOWNWASH_HEIGHT + (DOWNWASH_HIGH - DOWNWASH_HEIGHT) * cycle
    return x_offset, y_offset, z_height


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
# DRONE 1
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

    rms_norm      = 0.7+((rms_s-rms_s.min())/(rms_s.max()-rms_s.min()+1e-8))*0.7
    bass_norm     = 0.5+((bass_s-bass_s.min())/(bass_s.max()-bass_s.min()+1e-8))*1.0
    treb_norm     = 0.0+((treb_s-treb_s.min())/(treb_s.max()-treb_s.min()+1e-8))*0.3
    rms_long_norm = (rms_long-rms_long.min())/(rms_long.max()-rms_long.min()+1e-8)

    initPos    = cf.position()
    timesteps  = np.arange(0, SONG_END, 1/Hz)
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
        th    = get_value(t, freq_times, treb_norm)
        vocal = get_value(t, rms_times, rms_long_norm)

        x = D1_START[0]; y = D1_START[1]
        z = TAKEOFF_HEIGHT + th
        z_min, z_max = D1_Z_MIN, D1_Z_MAX

        # ===== BROKEN HEART =====
        if t >= HEART_START:
            hf = smooth_fade(t, HEART_START, duration=1.5)
            if t < HEART_BREAK:
                hx, hy = get_heart_half(t, 'left',
                                        start_time=HEART_START,
                                        duration=HEART_BREAK-HEART_START)
                x = blend(prev_x, FABRIC_CENTER_X+hx, hf)
                y = blend(prev_y, FABRIC_CENTER_Y+hy, hf)
            else:
                bp = min((t-HEART_BREAK)/3.0, 1.0)
                hx, hy = get_heart_half(t, 'left',
                                        start_time=HEART_START,
                                        duration=HEART_BREAK-HEART_START,
                                        break_offset=bp*0.35)
                x = FABRIC_CENTER_X+hx; y = FABRIC_CENTER_Y+hy
            z = 1.2; z_min, z_max = 1.0, 1.5

        # ===== SECTION 1 =====
        elif t <= S1_END:
            if t < CIRCLE1_START:
                sx, sy, sz = fabric_sweep(t, drone_id=1, sweep_speed=0.10)
                x = FABRIC_CENTER_X+sx; y = FABRIC_CENTER_Y+sy
                hop = BEAT_PULSE_HEIGHT*(0.4+vocal*0.6)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat, sz, hop, Hz)
            elif t <= CIRCLE1_END:
                c1_angle += speed*SPEED_SCALE*0.07
                cx, cy = rotated_circle(c1_angle, 0.6+vocal*0.5)
                x = D1_START[0]+cx; y = D1_START[1]+cy
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.5, Hz)
            elif t < CIRCLE2_START:
                sx, sy, sz = fabric_sweep(t, drone_id=1, sweep_speed=0.09)
                x = FABRIC_CENTER_X+sx; y = FABRIC_CENTER_Y+sy
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                sz, BEAT_PULSE_HEIGHT, Hz)
                if ENABLE_SPIRAL:
                    for dt in drop_times:
                        if abs(t-dt) < (1/Hz):
                            if (dt-last_spiral) >= SPIRAL_COOLDOWN:
                                spiral_start = t; last_spiral = t
                            break
                    tss = t-spiral_start
                    if 0 <= tss <= SPIRAL_DURATION:
                        spx, spy, spz = get_spiral_offset(tss)
                        x += spx; y += spy; z += spz
            elif t <= CIRCLE2_END:
                c2_angle += speed*SPEED_SCALE*0.06
                cx, cy = rotated_circle(c2_angle, 0.5+vocal*0.4)
                x = D1_START[0]+cx; y = D1_START[1]+cy
                z = 0.85+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.6, Hz)
            elif t <= TORNADO_END:
                tx, ty, tz = tornado_move(t, TORNADO_START,
                                          duration=TORNADO_END-TORNADO_START)
                fade = smooth_fade(t, TORNADO_START, duration=1.0)
                x = D1_START[0]+tx*fade; y = D1_START[1]+ty*fade
                z = FABRIC_HEIGHT+tz
            else:
                sx, sy, sz = fabric_sweep(t, drone_id=1, sweep_speed=0.08)
                x = FABRIC_CENTER_X+sx*0.5; y = FABRIC_CENTER_Y+sy*0.5; z = sz

        # ===== SECTION 2 =====
        elif t <= S2_END:
            z_min, z_max = 1.5, 2.5
            if t < BOTH_CIRCLE_START:
                s2_angle += speed*SPEED_SCALE*0.045
                x = D1_START[0]+1.1*np.sin(s2_angle)
                y = D1_START[1]+1.0*np.sin(2*s2_angle+np.pi/4)
                z = 1.8+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, Hz)
            elif t <= BOTH_CIRCLE_END:
                orb_angle += 1.1*SPEED_SCALE*(1/Hz)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                cx, cy = rotated_circle(orb_angle, 1.0)
                x = cx*fade; y = cy*fade; z = 1.8+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.7, Hz)
            elif t <= SWAY_END:
                fade = smooth_fade(t, SWAY_START, duration=1.5)
                beat_idx = int(np.searchsorted(beat_times, t)) % 4
                sway_dir = 1.0 if beat_idx < 2 else -1.0
                x = sway_dir*1.0*vocal*fade
                y = D1_START[1]*(1-fade*0.5); z = 1.8+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*1.1, Hz)
            else:
                orb_angle += 0.7*SPEED_SCALE*(1/Hz)
                cx, cy = rotated_circle(orb_angle, 1.0)
                x = cx; y = cy*0.8; z = 1.8+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*1.4, Hz)

        # ===== SECTION 3 =====
        else:
            z_min, z_max = D1_Z_MIN, D1_Z_MAX
            fade = smooth_fade(t, S2_END, duration=BLEND_DURATION)
            if t <= UPDOWN_END:
                orbit_s3 = 2*np.pi/8.0
                ox = FABRIC_X_HALF*np.sin(orbit_s3*(t-S2_END))
                oy = FABRIC_Y_HALF*np.cos(orbit_s3*(t-S2_END))
                x = FABRIC_CENTER_X+ox*fade; y = FABRIC_CENTER_Y+oy*fade
                z = 1.8+th+updown_alt(t, drone_id=1, amplitude=0.35)*fade
            else:
                fin_angle += 2*np.pi/9.0*(1/Hz)
                progress = (t-FINAL_CIRCLE_START)/(SONG_END-FINAL_CIRCLE_START)
                radius = (0.4+progress*1.2) if progress < 0.5 else 1.0*(1-(progress-0.5)/0.5)
                cx, cy = rotated_circle(fin_angle, radius)
                x = cx; y = cy; z = 1.8+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT, Hz)

        if abs(t-S1_END) < 2.0 and t > S1_END:
            bf = smooth_fade(t, S1_END, duration=2.0)
            x = blend(prev_x, x, bf); y = blend(prev_y, y, bf)
        if abs(t-S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf); y = blend(prev_y, y, bf)

        if t >= land_start:
            lp = min((t-land_start)/4.0, 1.0)
            x = blend(x, 0.0, lp); y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp); z_min, z_max = Z_MIN, Z_MAX

        prev_x = x; prev_y = y
        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 1 complete.")


# =========================================================
# DRONE 2
# =========================================================

def run_drone2(cf, features):
    rms_times  = features['rms_times']
    rms        = features['rms']
    bass       = features['bass_energy']
    treble     = features['treble_energy']
    freq_times = features['freq_times']
    beat_times_full = features['beat_times'][features['beat_times'] <= SONG_END]
    beat_times = beat_times_full[beat_times_full >= S1_END]

    rms_s    = smooth(rms,    window=60)
    bass_s   = smooth(bass,   window=60)
    treb_s   = smooth(treble, window=60)
    rms_long = smooth(rms,    window=200)

    rms_norm      = 0.7+((rms_s-rms_s.min())/(rms_s.max()-rms_s.min()+1e-8))*0.7
    bass_norm     = 0.5+((bass_s-bass_s.min())/(bass_s.max()-bass_s.min()+1e-8))*1.0
    treb_norm     = 0.0+((treb_s-treb_s.min())/(treb_s.max()-treb_s.min()+1e-8))*0.3
    rms_long_norm = (rms_long-rms_long.min())/(rms_long.max()-rms_long.min()+1e-8)

    initPos    = cf.position()
    timesteps  = np.arange(S1_END, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    orb_angle = np.pi
    fin_angle = 0.0
    prev_x = D2_START[0]
    prev_y = D2_START[1]

    for t in timesteps:
        speed = get_value(t, rms_times, rms_norm)
        th    = get_value(t, freq_times, treb_norm)
        vocal = get_value(t, rms_times, rms_long_norm)

        x = D2_START[0]; y = D2_START[1]
        z = 0.9+th
        z_min, z_max = D2_Z_MIN, D2_Z_MAX
        fade_in = smooth_fade(t, S1_END, duration=BLEND_DURATION)

        # ===== BROKEN HEART =====
        if t >= HEART_START:
            hf = smooth_fade(t, HEART_START, duration=1.5)
            if t < HEART_BREAK:
                hx, hy = get_heart_half(t, 'right',
                                        start_time=HEART_START,
                                        duration=HEART_BREAK-HEART_START)
                x = blend(prev_x, FABRIC_CENTER_X+hx, hf)
                y = blend(prev_y, FABRIC_CENTER_Y+hy, hf)
            else:
                bp = min((t-HEART_BREAK)/3.0, 1.0)
                hx, hy = get_heart_half(t, 'right',
                                        start_time=HEART_START,
                                        duration=HEART_BREAK-HEART_START,
                                        break_offset=bp*0.35)
                x = FABRIC_CENTER_X+hx; y = FABRIC_CENTER_Y+hy
            z = 1.2; z_min, z_max = 1.0, 1.5

        # ===== SECTION 2 =====
        elif t <= S2_END:
            if t < BOTH_CIRCLE_START:
                ts = t-ARC_START
                bx = 0.7*np.sin(1.8*SPEED_SCALE*ts)
                by = 1.1*np.cos(1.8*SPEED_SCALE*ts)
                x = D2_START[0]+bx*fade_in; y = D2_START[1]+by*fade_in
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.7, Hz)
            elif t <= BOTH_CIRCLE_END:
                orb_angle += 0.7*SPEED_SCALE*(1/Hz)
                fade = smooth_fade(t, BOTH_CIRCLE_START, duration=2.0)
                cx, cy = rotated_circle(orb_angle+np.pi, 1.0)
                x = cx*fade; y = cy*fade
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.7, Hz)
            elif t <= SWAY_END:
                fade = smooth_fade(t, SWAY_START, duration=1.5)
                beat_idx = int(np.searchsorted(beat_times, t)) % 4
                sway_dir = 1.0 if beat_idx < 2 else -1.0
                x = sway_dir*1.0*vocal*fade
                y = D2_START[1]*(1-fade*0.5)
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.9, Hz)
            else:
                helix_dur = D2_HELIX_END-D2_HELIX_START
                descent_start = D2_HELIX_END-5.0
                if t < descent_start:
                    hx, hy, hz = helix_move(t, D2_HELIX_START,
                                            duration=helix_dur, max_r=1.0, max_h=1.1)
                    x = D2_START[0]+hx; y = D2_START[1]+hy
                    z = 0.8+hz+th; z_min, z_max = D2_Z_MIN, 2.2
                else:
                    dp = (t-descent_start)/5.0
                    hx, hy, hz = helix_move(descent_start, D2_HELIX_START,
                                            duration=helix_dur, max_r=1.0, max_h=1.1)
                    x = blend(D2_START[0]+hx, D2_START[0], dp)
                    y = blend(D2_START[1]+hy, D2_START[1], dp)
                    z = blend(0.8+hz+th, 0.8, dp)
                    z_min, z_max = D2_Z_MIN, 2.2

        # ===== SECTION 3 =====
        else:
            z_min, z_max = Z_MIN, Z_MAX
            fade = smooth_fade(t, S2_END+2.0, duration=BLEND_DURATION)
            if t <= UPDOWN_END:
                orbit_s3 = 2*np.pi/8.0
                base_angle = 2*np.pi/3
                ox = FABRIC_X_HALF*np.sin(orbit_s3*(t-S2_END)+base_angle)
                oy = FABRIC_Y_HALF*np.cos(orbit_s3*(t-S2_END)+base_angle)
                sx, sy, sz = fabric_sweep(t, drone_id=2, sweep_speed=0.04)
                x = FABRIC_CENTER_X+ox*fade+sx*0.2*fade
                y = FABRIC_CENTER_Y+oy*fade+sy*0.2*fade
                z = 1.3+th+updown_alt(t, drone_id=2, amplitude=0.35)*fade
            else:
                fin_angle += 2*np.pi/9.0*(1/Hz)
                progress = (t-FINAL_CIRCLE_START)/(SONG_END-FINAL_CIRCLE_START)
                radius = (0.3+progress*0.8) if progress < 0.5 else 0.7*(1-(progress-0.5)/0.5)
                delay_angle = fin_angle-(1.5*2*np.pi/9.0)
                cx, cy = rotated_circle(delay_angle, radius)
                x = cx; y = cy; z = 1.3+th
                z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                                z, BEAT_PULSE_HEIGHT*0.8, Hz)

        if abs(t-S2_END) < 2.0 and t > S2_END:
            bf = smooth_fade(t, S2_END, duration=2.0)
            x = blend(prev_x, x, bf); y = blend(prev_y, y, bf)

        if t >= land_start:
            lp = min((t-land_start)/4.0, 1.0)
            x = blend(x, 0.0, lp); y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp); z_min, z_max = Z_MIN, Z_MAX

        prev_x = x; prev_y = y
        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        time.sleep(1/Hz)

    cf.notifySetpointsStop()
    print("Drone 2 complete.")


# =========================================================
# DRONE 3
# =========================================================

def run_drone3(cf, features):
    rms_times  = features['rms_times']
    rms        = features['rms']
    treble     = features['treble_energy']
    beat_times_full = features['beat_times'][features['beat_times'] <= SONG_END]
    beat_times = beat_times_full[beat_times_full >= S2_END]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= S2_END) & (drop_times <= SONG_END)]

    rms_s  = smooth(rms,    window=60)
    treb_s = smooth(treble, window=60)
    rms_norm  = 0.7+((rms_s-rms_s.min())/(rms_s.max()-rms_s.min()+1e-8))*0.7
    treb_norm = 0.0+((treb_s-treb_s.min())/(treb_s.max()-treb_s.min()+1e-8))*0.25

    initPos    = cf.position()
    timesteps  = np.arange(S2_END, SONG_END, 1/Hz)
    land_start = SONG_END - 4.0

    last_beat    = -BEAT_COOLDOWN
    last_spiral  = -SPIRAL_COOLDOWN
    spiral_start = -SPIRAL_COOLDOWN
    fin_angle    = 0.0
    prev_x = D3_START[0]
    prev_y = D3_START[1]

    for t in timesteps:
        th      = get_value(t, rms_times, treb_norm)
        fade_in = smooth_fade(t, S2_END+1.0, duration=BLEND_DURATION)

        x = D3_START[0]; y = D3_START[1]
        z = 0.8+th
        z_min, z_max = D3_Z_MIN, D3_Z_MAX

        # ===== BROKEN HEART — slow rise above center =====
        if t >= HEART_START:
            rise = smooth_fade(t, HEART_START, duration=4.0)
            x = blend(prev_x, FABRIC_CENTER_X, rise)
            y = blend(prev_y, FABRIC_CENTER_Y, rise)
            z = blend(DOWNWASH_HEIGHT, 2.2, rise)
            z_min, z_max = DOWNWASH_HEIGHT, 2.5

        elif t <= UPDOWN_END:
            orbit_s3 = 2*np.pi/8.0
            base_angle = 4*np.pi/3
            ox = FABRIC_X_HALF*np.sin(orbit_s3*(t-S2_END)+base_angle)
            oy = FABRIC_Y_HALF*np.cos(orbit_s3*(t-S2_END)+base_angle)
            sx, sy, sz = fabric_sweep(t, drone_id=3, sweep_speed=0.04)
            x = FABRIC_CENTER_X+ox*fade_in+sx*0.2*fade_in
            y = FABRIC_CENTER_Y+oy*fade_in+sy*0.2*fade_in
            z = DOWNWASH_HEIGHT+th+updown_alt(t, drone_id=3, amplitude=0.3)*fade_in

        else:
            fin_angle += 2*np.pi/9.0*(1/Hz)
            progress = (t-FINAL_CIRCLE_START)/(SONG_END-FINAL_CIRCLE_START)
            radius = (0.2+progress*0.6) if progress < 0.5 else 0.5*(1-(progress-0.5)/0.5)
            delay_angle = fin_angle-(3.0*2*np.pi/9.0)
            cx, cy = rotated_circle(delay_angle, radius)
            x = cx*fade_in; y = cy*fade_in
            z = DOWNWASH_HEIGHT+th; z_min, z_max = Z_MIN, Z_MAX

            z, last_beat = apply_beat_pulse(t, beat_times, last_beat,
                                            z, BEAT_PULSE_HEIGHT*0.7, Hz)
            if ENABLE_SPIRAL:
                for dt in drop_times:
                    if abs(t-dt) < (1/Hz):
                        if (dt-last_spiral) >= SPIRAL_COOLDOWN:
                            spiral_start = t; last_spiral = t
                        break
                tss = t-spiral_start
                if 0 <= tss <= SPIRAL_DURATION:
                    spx, spy, spz = get_spiral_offset(tss)
                    x += spx*0.3*fade_in; y += spy*0.3*fade_in; z += spz*0.3

        if t >= land_start:
            lp = min((t-land_start)/4.0, 1.0)
            x = blend(x, 0.0, lp); y = blend(y, 0.0, lp)
            z = blend(z, Z_MIN, lp); z_min, z_max = Z_MIN, Z_MAX

        prev_x = x; prev_y = y
        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
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
        print(f"Duration:  {features['duration']:.1f}s")
        print(f"Beats:     {len(features['beat_times'])}")
        print(f"\nSchedule:")
        print(f"  t=0s       D1 takeoff + choreography")
        print(f"  t={DRONE2_TAKEOFF_DELAY}s    D2 takeoff")
        print(f"  t={S1_END}s    D2 choreography")
        print(f"  t={DRONE3_TAKEOFF_DELAY}s   D3 takeoff")
        print(f"  t={S2_END}s   D3 choreography")
        print(f"  t={HEART_START}s  Broken heart begins")
        print(f"  t={HEART_BREAK}s  Heart breaks apart")
        print(f"  t={SONG_END}s All land")
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

    # Takeoff drone 1 and set LED once
    print("Drone 1 taking off...")
    cf1.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    time.sleep(TAKEOFF_DURATION + 1.0)
    set_led(cf1, D1_LED)

    perf_start = time.time()

    def drone1_thread():
        try:
            run_drone1(cf1, features)
        except Exception as e:
            print(f"D1 error: {e}")

    def drone2_thread():
        elapsed = time.time() - perf_start
        remaining = DRONE2_TAKEOFF_DELAY - elapsed
        if remaining > 0: time.sleep(remaining)
        if cf2 is None: print("No drone 2"); return
        print("Drone 2 taking off...")
        cf2.takeoff(targetHeight=DRONE2_TAKEOFF_HEIGHT,
                    duration=DRONE2_TAKEOFF_DURATION)
        time.sleep(DRONE2_TAKEOFF_DURATION + 1.5)
        set_led(cf2, D2_LED)   # set LED once after takeoff
        elapsed = time.time() - perf_start
        remaining = S1_END - elapsed
        if remaining > 0: time.sleep(remaining)
        print("Drone 2 choreography starting...")
        try:
            run_drone2(cf2, features)
        except Exception as e:
            print(f"D2 error: {e}")

    def drone3_thread():
        elapsed = time.time() - perf_start
        remaining = DRONE3_TAKEOFF_DELAY - elapsed
        if remaining > 0: time.sleep(remaining)
        if cf3 is None: print("No drone 3"); return
        print("Drone 3 taking off...")
        cf3.takeoff(targetHeight=DRONE3_TAKEOFF_HEIGHT,
                    duration=DRONE3_TAKEOFF_DURATION)
        time.sleep(DRONE3_TAKEOFF_DURATION + 1.5)
        set_led(cf3, D3_LED)   # set LED once after takeoff
        elapsed = time.time() - perf_start
        remaining = S2_END - elapsed
        if remaining > 0: time.sleep(remaining)
        print("Drone 3 choreography starting...")
        try:
            run_drone3(cf3, features)
        except Exception as e:
            print(f"D3 error: {e}")

    t1 = threading.Thread(target=drone1_thread)
    t2 = threading.Thread(target=drone2_thread)
    t3 = threading.Thread(target=drone3_thread)

    try:
        t1.start(); t2.start(); t3.start()
        t1.join();  t2.join();  t3.join()
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