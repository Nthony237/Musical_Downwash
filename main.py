import numpy as np
import time
from types import SimpleNamespace
from scipy.ndimage import uniform_filter1d
import sys
import os
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
DURATION = 111.0

# ---- SAFETY BOUNDS ----
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0
Z_MIN, Z_MAX = 0.5, 2.5

# ---- HEIGHT ZONES ----
D1_S1_Z_MIN, D1_S1_Z_MAX = 0.5, 2.5
D1_S2_Z_MIN, D1_S2_Z_MAX = 1.5, 2.5
D2_S2_Z_MIN, D2_S2_Z_MAX = 0.5, 1.4

# ---- BEAT SETTINGS ----
BEAT_COOLDOWN = 1.6
BEAT_PULSE_HEIGHT = 0.30
BEAT_PULSE_DURATION = 0.8

# ---- SPIRAL SETTINGS ----
SPIRAL_DURATION = 4.0
SPIRAL_COOLDOWN = 4.0
SPIRAL_HEIGHT = 1.0
SPIRAL_RADIUS_START = 0.1
SPIRAL_RADIUS_END = 1.8
ENABLE_SPIRAL = True
BLEND_DURATION = 5.0

# ---- INTERACTION SETTINGS ----
INTERACT_DURATION = 8.0
INTERACT_RADIUS = 1.0
MEET_HEIGHT = 1.5

# ---- DRONE 2 SETTINGS ----
DRONE2_START_TIME = 55.5      # when drone 2 joins the choreography
DRONE2_TAKEOFF_DELAY = 50.0   # seconds after start to begin drone 2 takeoff
DRONE2_TAKEOFF_HEIGHT = 0.8   # drone 2 hovers lower than drone 1
DRONE2_TAKEOFF_DURATION = 3.0

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
    if t_since_interact < 0 or t_since_interact > INTERACT_DURATION:
        return 0.0, 0.0, 0.0

    half = INTERACT_DURATION / 2.0

    if t_since_interact <= half:
        inward_progress = t_since_interact / half
        radius = INTERACT_RADIUS * (1 - inward_progress)
        spiral_angle = inward_progress * 4 * np.pi
        if drone_id == 1:
            z_offset = -(1.8 - MEET_HEIGHT) * inward_progress
        else:
            z_offset = (MEET_HEIGHT - 0.9) * inward_progress
    else:
        outward_progress = (t_since_interact - half) / half
        radius = INTERACT_RADIUS * outward_progress
        spiral_angle = outward_progress * 4 * np.pi + 4 * np.pi
        if drone_id == 1:
            z_offset = -(1.8 - MEET_HEIGHT) * (1 - outward_progress)
        else:
            z_offset = (MEET_HEIGHT - 0.9) * (1 - outward_progress)

    if drone_id == 2:
        spiral_angle += np.pi

    x_offset = radius * np.cos(spiral_angle)
    y_offset = radius * np.sin(spiral_angle)
    return x_offset, y_offset, z_offset

def find_biggest_drop(drop_times, onset_strength, rms_times, section_start, section_end):
    mask = (drop_times >= section_start) & (drop_times <= section_end)
    section_drops = drop_times[mask]
    if len(section_drops) == 0:
        return (section_start + section_end) / 2
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

def run_drone1(cf, timeHelper, features, interact_time):
    """Runs drone 1 choreography for the full duration."""
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    treble_energy = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[beat_times <= DURATION]
    drop_times = features['drop_times']
    drop_times = drop_times[drop_times <= DURATION]
    section_times = features['section_times']
    s1_end = section_times[1]

    rms_smooth = smooth(rms, window=60)
    bass_smooth = smooth(bass_energy, window=60)
    treble_smooth = smooth(treble_energy, window=60)

    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6

    bass_norm = (bass_smooth - bass_smooth.min()) / (bass_smooth.max() - bass_smooth.min())
    bass_norm = 0.5 + bass_norm * 1.0

    treble_norm = (treble_smooth - treble_smooth.min()) / (treble_smooth.max() - treble_smooth.min())
    treble_norm = 0.0 + treble_norm * 0.3

    initPos = cf.position()
    timesteps = np.arange(0, DURATION, 1/Hz)
    land_start = DURATION - 3.0

    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN

    for t in timesteps:
        speed = get_value(t, rms_times, rms_norm)
        size = get_value(t, freq_times, bass_norm)
        treble_height = get_value(t, freq_times, treble_norm)

        in_interaction = (t >= interact_time and
                         t <= interact_time + INTERACT_DURATION and
                         t >= s1_end)

        if t <= s1_end:
            angle += speed * (2 * np.pi / Hz) * 0.032
            x, y = figure8(angle, size)
            z = TAKEOFF_HEIGHT + treble_height
            z_min, z_max = D1_S1_Z_MIN, D1_S1_Z_MAX

            for bt in beat_times:
                time_since_beat = t - bt
                if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                    if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                        z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                        if time_since_beat < (1 / Hz):
                            last_beat_time = bt
                    break

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

        else:
            angle += speed * (2 * np.pi / Hz) * 0.032
            blend_factor = min((t - s1_end) / BLEND_DURATION, 1.0)
            f8x, f8y = figure8(angle, size)
            lsx, lsy = lissajous(angle, size)
            x = blend_shapes(f8x, lsx, blend_factor)
            y = blend_shapes(f8y, lsy, blend_factor)
            z = 1.8 + treble_height
            z_min, z_max = D1_S2_Z_MIN, D1_S2_Z_MAX

            if not in_interaction:
                for bt in beat_times:
                    time_since_beat = t - bt
                    if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                        if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                            z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT)
                            if time_since_beat < (1 / Hz):
                                last_beat_time = bt
                        break

                if ENABLE_SPIRAL:
                    for dt in drop_times:
                        if abs(t - dt) < (1 / Hz): 
                            if dt != interact_time:
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
                t_since_interact = t - interact_time
                ix, iy, iz = get_interaction_offset(t_since_interact, drone_id=1)
                x += ix
                y += iy
                z += iz
                z_min, z_max = Z_MIN, Z_MAX

        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = TAKEOFF_HEIGHT * (1 - land_progress) + Z_MIN * land_progress
            z_min, z_max = Z_MIN, Z_MAX

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        timeHelper.sleepForRate(Hz)

    cf.notifySetpointsStop()

def run_drone2(cf, timeHelper, features, interact_time):
    """Runs drone 2 choreography starting at DRONE2_START_TIME."""
    rms_times = features['rms_times']
    rms = features['rms']
    bass_energy = features['bass_energy']
    treble_energy = features['treble_energy']
    freq_times = features['freq_times']
    beat_times = features['beat_times']
    beat_times = beat_times[(beat_times >= DRONE2_START_TIME) & (beat_times <= DURATION)]
    drop_times = features['drop_times']
    drop_times = drop_times[(drop_times >= DRONE2_START_TIME) & (drop_times <= DURATION)]

    rms_smooth = smooth(rms, window=60)
    bass_smooth = smooth(bass_energy, window=60)
    treble_smooth = smooth(treble_energy, window=60)

    rms_norm = (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min())
    rms_norm = 0.7 + rms_norm * 0.6

    bass_norm = (bass_smooth - bass_smooth.min()) / (bass_smooth.max() - bass_smooth.min())
    bass_norm = 0.5 + bass_norm * 1.0

    treble_norm = (treble_smooth - treble_smooth.min()) / (treble_smooth.max() - treble_smooth.min())
    treble_norm = 0.0 + treble_norm * 0.2

    initPos = cf.position()
    # Drone 2 only runs from its start time to end
    timesteps = np.arange(DRONE2_START_TIME, DURATION, 1/Hz)
    land_start = DURATION - 3.0

    angle = 0.0
    last_beat_time = -BEAT_COOLDOWN
    last_spiral_time = -SPIRAL_COOLDOWN
    current_spiral_start = -SPIRAL_COOLDOWN

    for t in timesteps:
        speed = get_value(t, rms_times, rms_norm)
        size = get_value(t, freq_times, bass_norm)
        treble_height = get_value(t, freq_times, treble_norm)

        in_interaction = (t >= interact_time and
                         t <= interact_time + INTERACT_DURATION)

        angle -= speed * (2 * np.pi / Hz) * 0.032
        x, y = figure8(angle, size)
        z = 0.9 + treble_height
        z_min, z_max = D2_S2_Z_MIN, D2_S2_Z_MAX

        fade_in = min((t - DRONE2_START_TIME) / BLEND_DURATION, 1.0)
        x *= fade_in
        y *= fade_in

        if not in_interaction:
            for bt in beat_times:
                time_since_beat = t - bt
                if 0 <= time_since_beat <= BEAT_PULSE_DURATION:
                    if (bt - last_beat_time) >= BEAT_COOLDOWN or bt == last_beat_time:
                        z += get_beat_pulse(time_since_beat, BEAT_PULSE_DURATION, BEAT_PULSE_HEIGHT * 0.8)
                        if time_since_beat < (1 / Hz):
                            last_beat_time = bt
                    break

            if ENABLE_SPIRAL:
                for dt in drop_times:
                    if abs(t - dt) < (1 / Hz):
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
            t_since_interact = t - interact_time
            ix, iy, iz = get_interaction_offset(t_since_interact, drone_id=2)
            x += ix
            y += iy
            z += iz
            z_min, z_max = Z_MIN, Z_MAX

        if t >= land_start:
            land_progress = (t - land_start) / 3.0
            x = x * (1 - land_progress)
            y = y * (1 - land_progress)
            z = 0.9 * (1 - land_progress) + Z_MIN * land_progress
            z_min, z_max = Z_MIN, Z_MAX

        position = np.array([x, y, z]) + np.array(initPos)
        position = clamp_position(position, z_min, z_max)
        cf.cmdPosition(position)
        timeHelper.sleepForRate(Hz)

    cf.notifySetpointsStop()

def emergency_stop(crazyflies):
    print("\nEMERGENCY STOP")
    for cf in crazyflies:
        cf.notifySetpointsStop()
        cf.land(targetHeight=0.04, duration=2.0)

def main():
    global SIM, DRY_RUN

    if DRY_RUN:
        print("DRY RUN - printing positions only")
        cf1 = SimpleNamespace(
            position=lambda: [0, 0, 0],
            cmdPosition=lambda p: None,
            notifySetpointsStop=lambda: None,
            takeoff=lambda **k: None,
            land=lambda **k: None
        )
        cf2 = SimpleNamespace(
            position=lambda: [0.5, 0, 0],  # offset so they don't start same spot
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
            crazyflies=[cf1, cf2],
            timeHelper=timeHelper
        )

        print("Extracting audio features...")
        features = extract_features(AUDIO_PATH)
        onset_strength = features.get('onset_strength', features['rms'])
        interact_time = find_biggest_drop(
            features['drop_times'], onset_strength,
            features['rms_times'],
            features['section_times'][1],
            features['section_times'][2]
        )
        print(f"Interaction at t={interact_time:.1f}s")
        print("Dry run complete.")
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

    if len(crazyflies) < 2:
        print("WARNING: Less than 2 drones detected. Running drone 1 only.")

    cf1 = crazyflies[0]
    cf2 = crazyflies[1] if len(crazyflies) > 1 else None

    print("Extracting audio features...")
    features = extract_features(AUDIO_PATH)
    onset_strength = features.get('onset_strength', features['rms'])
    interact_time = find_biggest_drop(
        features['drop_times'], onset_strength,
        features['rms_times'],
        features['section_times'][1],
        features['section_times'][2]
    )
    print(f"Interaction will happen at t={interact_time:.1f}s")

    # ---- DRONE 1 TAKEOFF ----
    print(f"Drone 1 taking off to {TAKEOFF_HEIGHT}m...")
    cf1.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)

    # ---- START DRONE 1 CHOREOGRAPHY IN BACKGROUND ----
    # Drone 1 runs the full duration
    # Drone 2 takes off at DRONE2_TAKEOFF_DELAY then joins

    import threading

    drone1_done = threading.Event()

    def drone1_thread():
        try:
            run_drone1(cf1, timeHelper, features, interact_time)
        except Exception as e:
            print(f"Drone 1 error: {e}")
        finally:
            drone1_done.set()

    def drone2_thread():
        # Wait until DRONE2_TAKEOFF_DELAY seconds into the performance
        print(f"Drone 2 waiting until t={DRONE2_TAKEOFF_DELAY}s to take off...")
        timeHelper.sleep(DRONE2_TAKEOFF_DELAY)

        if cf2 is None:
            print("No drone 2 available, skipping.")
            return

        # Drone 1 is in upper zone by now so safe to take off low
        print(f"Drone 2 taking off to {DRONE2_TAKEOFF_HEIGHT}m...")
        cf2.takeoff(targetHeight=DRONE2_TAKEOFF_HEIGHT, duration=DRONE2_TAKEOFF_DURATION)
        timeHelper.sleep(DRONE2_TAKEOFF_DURATION + 1.5)  # extra stabilization time

        print("Drone 2 starting choreography...")
        try:
            run_drone2(cf2, timeHelper, features, interact_time)
        except Exception as e:
            print(f"Drone 2 error: {e}")

    # Start both threads
    t1 = threading.Thread(target=drone1_thread)
    t2 = threading.Thread(target=drone2_thread)

    try:
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        emergency_stop(crazyflies)
        return

    # ---- LAND BOTH ----
    print("Landing all drones...")
    for cf in crazyflies:
        cf.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 1.0)
    print("Landed successfully.")

if __name__ == '__main__':
    main()