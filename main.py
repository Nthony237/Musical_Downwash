import numpy as np
import time
from types import SimpleNamespace
from scipy.ndimage import uniform_filter1d
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
SIM = False
DRY_RUN = False
Hz = 20
AUDIO_PATH = 'audio/robots_mixdown.mp3'
TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 3.0
LAND_DURATION = 2.5
DURATION = 40.0

# ---- SAFETY BOUNDS ----
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.5, 1.5
Z_MIN, Z_MAX = 0.5, 2.0

# ---- BEAT SETTINGS ----
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.15
BEAT_PULSE_DURATION = 0.8

# ---- SPIRAL SETTINGS ----
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 4.0
SPIRAL_HEIGHT = 0.5
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 0.8
ENABLE_SPIRAL = True

def clamp_position(position):
    x = np.clip(position[0], X_MIN, X_MAX)
    y = np.clip(position[1], Y_MIN, Y_MAX)
    z = np.clip(position[2], Z_MIN, Z_MAX)
    return np.array([x, y, z])

def smooth(signal, window=60):
    return uniform_filter1d(signal.astype(float), size=window)

def get_value(t, times, values):
    idx = np.searchsorted(times, t)
    return values[min(idx, len(values) - 1)]

def get_beat_pulse(t_since_beat, duration=0.8, height=0.15):
    if t_since_beat < 0 or t_since_beat > duration:
        return 0.0
    progress = t_since_beat / duration
    return height * np.sin(progress * np.pi)

def get_spiral_offset(t_since_drop):
    if t_since_drop < 0 or t_since_drop > SPIRAL_DURATION:
        return 0.0, 0.0, 0.0
    progress = t_since_drop / SPIRAL_DURATION
    spiral_angle = progress * 6 * np.pi
    z_offset = SPIRAL_HEIGHT * np.sin(progress * np.pi)
    if progress < 0.8:
        radius = SPIRAL_RADIUS_START + (SPIRAL_RADIUS_END - SPIRAL_RADIUS_START) * (progress / 0.8)
    else:
        fade = (progress - 0.8) / 0.2
        radius = SPIRAL_RADIUS_END * (1 - fade)
    x_offset = radius * np.cos(spiral_angle)
    y_offset = radius * np.sin(spiral_angle)
    return x_offset, y_offset, z_offset

def run_choreography(groupState):
    crazyflies = groupState.crazyflies
    timeHelper = groupState.timeHelper

    print("Extracting audio features...")
    features = extract_features(AUDIO_PATH)

    rms_smooth = smooth(features['rms'], window=60)
    bass_smooth = smooth(features['bass_energy'], window=60)
    rms_times = features['rms_times']
    freq_times = features['freq_times']

    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6

    bass_norm = (bass_smooth - bass_smooth.min()) / (bass_smooth.max() - bass_smooth.min())
    bass_norm = 0.5 + bass_norm * 1.0

    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= DURATION]

    drop_times = features['drop_times']
    drop_times = drop_times[drop_times <= DURATION]

    cf = crazyflies[0]
    initPos = cf.position()

    timesteps = np.arange(0, DURATION, 1/Hz)
    land_start = DURATION - 3.0

    print("Starting choreography...")

    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN

    for t in timesteps:
        speed = get_value(t, rms_times, rms_norm)
        size = get_value(t, freq_times, bass_norm)

        angle += speed * (2 * np.pi / Hz) * 0.032

        x = size * np.sin(angle)
        y = size * np.sin(angle) * np.cos(angle)
        z = TAKEOFF_HEIGHT

        # Beat pulse
        for bt in beat_times:
            time_since_beat = t - bt
            if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                    z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                    if time_since_beat < (1 / Hz):
                        last_beat_time = bt
                break

        # Spiral on beat drops
        if ENABLE_SPIRAL:
            for dt in drop_times:
                if abs(t - dt) < (1 / Hz):
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

        # Landing approach
        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = TAKEOFF_HEIGHT * (1 - land_progress) + Z_MIN * land_progress

        position = np.array([x, y, z])
        position = position + np.array(initPos)
        position = clamp_position(position)

        if DRY_RUN:
            print(f"t={t:.2f}s | pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) | speed={speed:.2f} | size={size:.2f}")
        else:
            cf.cmdPosition(position)
            timeHelper.sleepForRate(Hz)

    cf.notifySetpointsStop()
    print("Choreography complete.")

def emergency_stop(crazyflies):
    print("\nEMERGENCY STOP")
    for cf in crazyflies:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.04, duration=2.0)

def main():
    global SIM, DRY_RUN

    if DRY_RUN:
        print("DRY RUN - printing positions only")
        cf = SimpleNamespace(
            position=lambda: [0, 0, 0],
            cmdPosition=lambda p: None,
            notifySetpointsStop=lambda: None,
            takeoff=lambda **k: None,
            land=lambda **k: None
        )
        timeHelper = SimpleNamespace(
            sleepForRate=lambda hz: None,
            sleep=lambda t: None
        )
        groupState = SimpleNamespace(
            crazyflies=[cf],
            timeHelper=timeHelper
        )
        run_choreography(groupState)
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
    groupState = SimpleNamespace(crazyflies=crazyflies, timeHelper=timeHelper)

    print(f"Taking off to {TAKEOFF_HEIGHT}m...")
    for cf in crazyflies:
        cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)

    try:
        run_choreography(groupState)
    except KeyboardInterrupt:
        emergency_stop(crazyflies)

    print("Landing...")
    for cf in crazyflies:
        cf.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 1.0)
    print("Landed successfully.")

if __name__ == '__main__':
    main()