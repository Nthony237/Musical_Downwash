import numpy as np
import time
from types import SimpleNamespace
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features

# ---- CONFIG ----
SIM = True
DRY_RUN = True   # set to False when running on lab machine
Hz = 20
AUDIO_PATH = 'audio/robots_mixdown.mp3'
TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 3.0
LAND_DURATION = 2.5
DURATION = 40.0

# ---- SAFETY BOUNDS ----
X_MIN, X_MAX = -2, 2
Y_MIN, Y_MAX = -2, 2
Z_MIN, Z_MAX = 0.5, 2

# ---- BEAT SETTINGS ----
BEAT_COOLDOWN = 0.6       # seconds between beat pulses
BEAT_PULSE_HEIGHT = 0.4   # 40cm pulse on each beat

def clamp_position(position):
    """Clamp position to safe flying bounds."""
    x = np.clip(position[0], X_MIN, X_MAX)
    y = np.clip(position[1], Y_MIN, Y_MAX)
    z = np.clip(position[2], Z_MIN, Z_MAX)
    return np.array([x, y, z])

def get_speed(t, rms_times, rms_norm):
    idx = np.searchsorted(rms_times, t)
    return rms_norm[min(idx, len(rms_norm) - 1)]

def get_size(t, freq_times, bass_norm):
    idx = np.searchsorted(freq_times, t)
    return bass_norm[min(idx, len(bass_norm) - 1)]

def run_choreography(groupState):
    crazyflies = groupState.crazyflies
    timeHelper = groupState.timeHelper

    print("Extracting audio features...")
    features = extract_features(AUDIO_PATH)

    rms = features['rms']
    rms_times = features['rms_times']
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min())
    rms_norm = 0.5 + rms_norm * 2.5

    bass = features['bass_energy']
    freq_times = features['freq_times']
    bass_norm = (bass - bass.min()) / (bass.max() - bass.min())
    bass_norm = 0.5 + bass_norm * 1.0

    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= DURATION]

    cf = crazyflies[0]
    initPos = cf.position()

    timesteps = np.arange(0, DURATION, 1/Hz)

    print("Starting choreography...")
    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN

    for t in timesteps:
        speed = get_speed(t, rms_times, rms_norm)
        size = get_size(t, freq_times, bass_norm)

        angle += speed * (2 * np.pi / Hz) * 0.15

        x = size * np.sin(angle)
        y = size * np.sin(angle) * np.cos(angle)
        z = TAKEOFF_HEIGHT

        # Beat pulse with cooldown and graceful decay
        for bt in beat_times:
            time_since_beat = t - bt
            if 0 <= time_since_beat < (4 / Hz):
                if (bt - last_beat_time) >= BEAT_COOLDOWN:
                    z += BEAT_PULSE_HEIGHT * np.exp(-0.15 * (time_since_beat * Hz))
                    last_beat_time = bt
                break

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

if __name__ == '__main__':
    main()