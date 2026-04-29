import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pygame
import time
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features
from flight.choreography import (get_drone1_positions, get_drone2_positions,
                                  get_drone3_positions, apply_collision_avoidance,
                                  D1_START, D2_START, D3_START,
                                  SONG_END, S1_END, S2_END,
                                  CIRCLE1_START, CIRCLE1_END,
                                  CIRCLE2_START, CIRCLE2_END,
                                  TORNADO_START, TORNADO_END,
                                  BOTH_CIRCLE_START, BOTH_CIRCLE_END,
                                  SWAY_START, SWAY_END,
                                  D2_SPIRAL_START, UPDOWN_START,
                                  UPDOWN_END, FINAL_CIRCLE_START)

# Section label lookup
def get_section_label(t):
    if t < CIRCLE1_START:
        return "S1: BPM hops"
    elif t < CIRCLE1_END:
        return "S1: Circle w/ lyrics"
    elif t < CIRCLE2_START:
        return "S1: BPM + spirals"
    elif t < CIRCLE2_END:
        return "S1: Second circle (lower)"
    elif t < TORNADO_END:
        return "S1: TORNADO"
    elif t < BOTH_CIRCLE_START:
        return "S2: Lissajous + violin bow"
    elif t < BOTH_CIRCLE_END:
        return "S2: Both circle (asymmetric)"
    elif t < SWAY_END:
        return "S2: Lateral sway"
    elif t < S2_END:
        return "S2: D1 BPM + D2 helix"
    elif t < UPDOWN_END:
        return "S3: Up/down + pinwheel"
    elif t < FINAL_CIRCLE_START:
        return "S3: Final circle"
    else:
        return "S3: Converge + land"

def animate_choreography(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)

    features = extract_features(filepath)
    duration = SONG_END
    fps = 30
    total_frames = int(duration * fps)
    trail_length = 60

    print("Computing drone 1 positions...")
    x1, y1, z1 = get_drone1_positions(features, duration, fps)
    print("Computing drone 2 positions...")
    x2, y2, z2, d2_start = get_drone2_positions(features, duration, fps)
    print("Computing drone 3 positions...")
    x3, y3, z3, d3_start = get_drone3_positions(features, duration, fps)

    print("Applying collision avoidance...")
    corrected = apply_collision_avoidance(
        [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
    )
    x1, y1, z1 = corrected[0]
    x2, y2, z2 = corrected[1]
    x3, y3, z3 = corrected[2]
    print("Done. Starting animation...")

    # Print min distances for safety check
    d12 = np.sqrt((x1[d2_start:] - x2[d2_start:])**2 +
                  (y1[d2_start:] - y2[d2_start:])**2 +
                  (z1[d2_start:] - z2[d2_start:])**2)
    print(f"Min D1-D2 distance: {d12.min():.3f}m")
    d13 = np.sqrt((x1[d3_start:] - x3[d3_start:])**2 +
                  (y1[d3_start:] - y3[d3_start:])**2 +
                  (z1[d3_start:] - z3[d3_start:])**2)
    d23 = np.sqrt((x2[d3_start:] - x3[d3_start:])**2 +
                  (y2[d3_start:] - y3[d3_start:])**2 +
                  (z2[d3_start:] - z3[d3_start:])**2)
    print(f"Min D1-D3 distance: {d13.min():.3f}m")
    print(f"Min D2-D3 distance: {d23.min():.3f}m")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height (m)')
    ax.set_title('Musical Downwash — Full Song Choreography')

    # Start markers
    ax.scatter(*D1_START, color='cyan',    s=120, marker='^', alpha=0.6, zorder=5)
    ax.scatter(*D2_START, color='magenta', s=120, marker='^', alpha=0.6, zorder=5)
    ax.scatter(*D3_START, color='yellow',  s=120, marker='^', alpha=0.6, zorder=5)

    drone1, = ax.plot([], [], [], 'o', color='cyan',    markersize=12, label='Drone 1')
    trail1, = ax.plot([], [], [], '-', color='cyan',    alpha=0.4, linewidth=1.5)
    drone2, = ax.plot([], [], [], 'o', color='magenta', markersize=12, label='Drone 2')
    trail2, = ax.plot([], [], [], '-', color='magenta', alpha=0.4, linewidth=1.5)
    drone3, = ax.plot([], [], [], 'o', color='yellow',  markersize=12, label='Drone 3')
    trail3, = ax.plot([], [], [], '-', color='yellow',  alpha=0.4, linewidth=1.5)

    time_text    = ax.text2D(0.05, 0.97, '', transform=ax.transAxes, fontsize=9)
    height_text  = ax.text2D(0.05, 0.91, '', transform=ax.transAxes, fontsize=9)
    section_text = ax.text2D(0.05, 0.85, '', transform=ax.transAxes, fontsize=9,
                             color='white',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    dist_text    = ax.text2D(0.05, 0.79, '', transform=ax.transAxes,
                             fontsize=9, color='lime')
    ax.legend(loc='upper right', fontsize=9)

    # Safety planes
    xx, yy = np.meshgrid([-2.0, 2.0], [-2.0, 2.0])
    ax.plot_surface(xx, yy, np.full_like(xx, 0.5), alpha=0.04, color='red')
    ax.plot_surface(xx, yy, np.full_like(xx, 2.5), alpha=0.04, color='blue')
    # Height zone divider
    ax.plot_surface(xx, yy, np.full_like(xx, 1.5), alpha=0.03, color='yellow')

    start_wall = [None]

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

        # Drone 1 always visible
        drone1.set_data([x1[frame]], [y1[frame]])
        drone1.set_3d_properties([z1[frame]])
        s = max(0, frame - trail_length)
        trail1.set_data(x1[s:frame], y1[s:frame])
        trail1.set_3d_properties(z1[s:frame])

        # Drone 2
        if frame >= d2_start:
            drone2.set_data([x2[frame]], [y2[frame]])
            drone2.set_3d_properties([z2[frame]])
            s2 = max(d2_start, frame - trail_length)
            trail2.set_data(x2[s2:frame], y2[s2:frame])
            trail2.set_3d_properties(z2[s2:frame])
        else:
            drone2.set_data([], [])
            drone2.set_3d_properties([])
            trail2.set_data([], [])
            trail2.set_3d_properties([])

        # Drone 3
        if frame >= d3_start:
            drone3.set_data([x3[frame]], [y3[frame]])
            drone3.set_3d_properties([z3[frame]])
            s3 = max(d3_start, frame - trail_length)
            trail3.set_data(x3[s3:frame], y3[s3:frame])
            trail3.set_3d_properties(z3[s3:frame])
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
            d12 = np.linalg.norm(p1 - p2)
            dist_str = f"D1-D2:{d12:.2f}m"
            if d12 < 0.35:
                warning = True
        if frame >= d3_start:
            p3 = np.array([x3[frame], y3[frame], z3[frame]])
            d13 = np.linalg.norm(p1 - p3)
            d23 = np.linalg.norm(p2 - p3)
            dist_str += f" D1-D3:{d13:.2f}m D2-D3:{d23:.2f}m"
            if min(d13, d23) < 0.35:
                warning = True

        if warning:
            dist_str = "⚠ " + dist_str
            dist_text.set_color('red')
        else:
            dist_text.set_color('lime')

        time_text.set_text(f'Time: {t:.1f}s / {duration:.0f}s')
        height_text.set_text(
            f'D1:{z1[frame]:.2f}m  D2:{z2[frame]:.2f}m  D3:{z3[frame]:.2f}m')
        section_text.set_text(get_section_label(t))
        dist_text.set_text(dist_str)

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