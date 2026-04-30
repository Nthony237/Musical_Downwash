import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pygame
import time
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features
from flight.choreography import (
    get_drone1_positions, get_drone2_positions, get_drone3_positions,
    apply_collision_avoidance,
    D1_START, D2_START, D3_START, D1_LAND, D2_LAND, D3_LAND,
    SONG_END, S1_END, S2_END,
    CIRCLE1_START, CIRCLE1_END, CIRCLE2_START, CIRCLE2_END,
    TORNADO_START, TORNADO_END, BOTH_CIRCLE_START, BOTH_CIRCLE_END,
    SWAY_START, SWAY_END, D2_SPIRAL_START, D2_SPIRAL_END,
    UPDOWN_START, UPDOWN_END, FINAL_CIRCLE_START, LINE_START, LINE_END,
    X_MIN, X_MAX, Y_MIN, Y_MAX
)

def get_section_label(t):
    if t < CIRCLE1_START:      return "S1: Beat hops + drift"
    elif t < CIRCLE1_END:      return "S1: Circle w/ lyrics"
    elif t < CIRCLE2_START:    return "S1: BPM + spirals + drift"
    elif t < CIRCLE2_END:      return "S1: Second circle (lower)"
    elif t < TORNADO_END:      return "S1: TORNADO"
    elif t < LINE_START:       return "S2: Lissajous + violin bow"
    elif t < LINE_END:         return "S2: LINE FORMATION ←→"
    elif t < BOTH_CIRCLE_END:  return "S2: Both circle (asymmetric)"
    elif t < SWAY_END:         return "S2: Lateral sway"
    elif t < S2_END:           return "S2: D1 orbit + D2 helix"
    elif t < UPDOWN_END:       return "S3: Up/down + pinwheel"
    elif t < FINAL_CIRCLE_START: return "S3: Transition"
    else:                      return "S3: Final concentric circles"

def animate_choreography(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)

    features = extract_features(filepath)
    duration = SONG_END
    fps = 30
    total_frames = int(duration * fps)
    trail_length = 50

    print("Computing drone 1...")
    x1, y1, z1, c1 = get_drone1_positions(features, duration, fps)
    print("Computing drone 2...")
    x2, y2, z2, d2_start, c2 = get_drone2_positions(features, duration, fps)
    print("Computing drone 3...")
    x3, y3, z3, d3_start, c3 = get_drone3_positions(features, duration, fps)

    print("Applying collision avoidance...")
    corrected = apply_collision_avoidance(
        [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
    )
    x1, y1, z1 = corrected[0]
    x2, y2, z2 = corrected[1]
    x3, y3, z3 = corrected[2]

    # Print safety stats
    d12 = np.sqrt((x1[d2_start:] - x2[d2_start:])**2 +
                  (y1[d2_start:] - y2[d2_start:])**2 +
                  (z1[d2_start:] - z2[d2_start:])**2)
    d13 = np.sqrt((x1[d3_start:] - x3[d3_start:])**2 +
                  (y1[d3_start:] - y3[d3_start:])**2 +
                  (z1[d3_start:] - z3[d3_start:])**2)
    d23 = np.sqrt((x2[d3_start:] - x3[d3_start:])**2 +
                  (y2[d3_start:] - y3[d3_start:])**2 +
                  (z2[d3_start:] - z3[d3_start:])**2)
    print(f"Min distances — D1-D2: {d12.min():.3f}m  "
          f"D1-D3: {d13.min():.3f}m  D2-D3: {d23.min():.3f}m")

    # Velocity check
    dt = 1/fps
    for name, xp, yp, zp in [("D1", x1, y1, z1),
                               ("D2", x2, y2, z2),
                               ("D3", x3, y3, z3)]:
        spd = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2 + np.diff(zp)**2) / dt
        print(f"{name} max speed: {spd.max():.2f} m/s  "
              f"frames >1.0m/s: {(spd>1.0).sum()}  "
              f"frames >1.5m/s: {(spd>1.5).sum()}")

    print("Done. Starting animation...")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(0, 2.5)
    ax.set_xlabel('X (3m)')
    ax.set_ylabel('Y (2m)')
    ax.set_zlabel('Height (m)')
    ax.set_title('Musical Downwash — Full Choreography')

    # Draw structure boundary
    sx = [X_MIN, X_MAX, X_MAX, X_MIN, X_MIN]
    sy = [Y_MIN, Y_MIN, Y_MAX, Y_MAX, Y_MIN]
    ax.plot(sx, sy, [0.5]*5, 'w--', alpha=0.3, linewidth=1)
    ax.plot(sx, sy, [2.2]*5, 'w--', alpha=0.3, linewidth=1)

    # Start markers
    ax.scatter(*D1_START[:2], D1_START[2]+0.05,
               color='white', s=80, marker='^', alpha=0.5)
    ax.scatter(*D2_START[:2], D2_START[2]+0.05,
               color='white', s=80, marker='^', alpha=0.5)
    ax.scatter(*D3_START[:2], D3_START[2]+0.05,
               color='white', s=80, marker='^', alpha=0.5)

    drone1, = ax.plot([], [], [], 'o', markersize=14)
    trail1, = ax.plot([], [], [], '-', alpha=0.3, linewidth=1.5)
    drone2, = ax.plot([], [], [], 'o', markersize=14)
    trail2, = ax.plot([], [], [], '-', alpha=0.3, linewidth=1.5)
    drone3, = ax.plot([], [], [], 'o', markersize=14)
    trail3, = ax.plot([], [], [], '-', alpha=0.3, linewidth=1.5)

    time_text    = ax.text2D(0.05, 0.97, '', transform=ax.transAxes, fontsize=9)
    height_text  = ax.text2D(0.05, 0.91, '', transform=ax.transAxes, fontsize=9)
    section_text = ax.text2D(0.05, 0.85, '', transform=ax.transAxes, fontsize=9,
                             color='white',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    dist_text    = ax.text2D(0.05, 0.79, '', transform=ax.transAxes, fontsize=8)
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.title.set_color('white')

    start_wall = [None]

    def to_mpl_color(rgb_tuple):
        return (rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255)

    def init():
        pygame.mixer.music.play()
        start_wall[0] = time.time()
        for obj in [drone1, trail1, drone2, trail2, drone3, trail3]:
            obj.set_data([], [])
            obj.set_3d_properties([])
        return (drone1, trail1, drone2, trail2, drone3, trail3,
                time_text, height_text, section_text, dist_text)

    def update(i):
        if start_wall[0] is None:
            return (drone1, trail1, drone2, trail2, drone3, trail3,
                    time_text, height_text, section_text, dist_text)

        t = time.time() - start_wall[0]
        if t >= duration:
            pygame.mixer.music.stop()
            plt.close()
            return (drone1, trail1, drone2, trail2, drone3, trail3,
                    time_text, height_text, section_text, dist_text)

        frame = min(int(t * fps), total_frames - 1)

        # Get colors for this frame
        col1 = to_mpl_color(c1[frame])
        col2 = to_mpl_color(c2[frame]) if frame < len(c2) else (1,0,1)
        col3 = to_mpl_color(c3[frame]) if frame < len(c3) else (0,1,1)

        # Drone 1
        drone1.set_data([x1[frame]], [y1[frame]])
        drone1.set_3d_properties([z1[frame]])
        drone1.set_color(col1)
        s = max(0, frame - trail_length)
        trail1.set_data(x1[s:frame], y1[s:frame])
        trail1.set_3d_properties(z1[s:frame])
        trail1.set_color(col1)

        # Drone 2
        if frame >= d2_start:
            drone2.set_data([x2[frame]], [y2[frame]])
            drone2.set_3d_properties([z2[frame]])
            drone2.set_color(col2)
            s2 = max(d2_start, frame - trail_length)
            trail2.set_data(x2[s2:frame], y2[s2:frame])
            trail2.set_3d_properties(z2[s2:frame])
            trail2.set_color(col2)
        else:
            drone2.set_data([], [])
            drone2.set_3d_properties([])
            trail2.set_data([], [])
            trail2.set_3d_properties([])

        # Drone 3
        if frame >= d3_start:
            drone3.set_data([x3[frame]], [y3[frame]])
            drone3.set_3d_properties([z3[frame]])
            drone3.set_color(col3)
            s3 = max(d3_start, frame - trail_length)
            trail3.set_data(x3[s3:frame], y3[s3:frame])
            trail3.set_3d_properties(z3[s3:frame])
            trail3.set_color(col3)
        else:
            drone3.set_data([], [])
            drone3.set_3d_properties([])
            trail3.set_data([], [])
            trail3.set_3d_properties([])

        # Live distance check
        dist_str = ""
        warning = False
        if frame >= d2_start:
            p1 = np.array([x1[frame], y1[frame], z1[frame]])
            p2 = np.array([x2[frame], y2[frame], z2[frame]])
            d12v = np.linalg.norm(p1 - p2)
            dist_str = f"D1-D2:{d12v:.2f}m"
            if d12v < 0.35: warning = True
        if frame >= d3_start:
            p3 = np.array([x3[frame], y3[frame], z3[frame]])
            d13v = np.linalg.norm(p1 - p3)
            d23v = np.linalg.norm(p2 - p3)
            dist_str += f" D1-D3:{d13v:.2f}m D2-D3:{d23v:.2f}m"
            if min(d13v, d23v) < 0.35: warning = True

        dist_text.set_text(dist_str)
        dist_text.set_color('red' if warning else 'lime')

        time_text.set_text(f'Time: {t:.1f}s / {duration:.0f}s')
        height_text.set_text(
            f'D1:{z1[frame]:.2f}m  D2:{z2[frame]:.2f}m  D3:{z3[frame]:.2f}m')
        section_text.set_text(get_section_label(t))

        return (drone1, trail1, drone2, trail2, drone3, trail3,
                time_text, height_text, section_text, dist_text)

    ani = animation.FuncAnimation(
        fig, update,
        frames=iter(range(10**6)),
        init_func=init,
        interval=16,
        blit=True,
        repeat=False
    )
    plt.tight_layout()
    plt.show()
    pygame.mixer.music.stop()

if __name__ == "__main__":
    print("Starting visualization...")
    animate_choreography('../audio/robots_mixdown.mp3')