import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pygame
import time
import sys
sys.path.append('..')
from analysis.beat_analysis import extract_features
from flight.choreography import get_drone1_positions, get_drone2_positions

def animate_choreography(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)

    features = extract_features(filepath)

    duration = 111.0
    fps = 30
    total_frames = int(duration * fps)
    trail_length = 45

    print(f"Computing {duration:.0f}s of positions ({total_frames} frames)...")
    x1, y1, z1 = get_drone1_positions(features, duration, fps)
    x2, y2, z2, drone2_start = get_drone2_positions(features, duration, fps, start_time=55.5)
    print("Done. Starting animation...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height (m)')
    ax.set_title('Musical Downwash - Two Drone Choreography')

    drone1, = ax.plot([], [], [], 'o', color='cyan', markersize=12, label='Drone 1')
    trail1, = ax.plot([], [], [], '-', color='cyan', alpha=0.3, linewidth=1)
    drone2, = ax.plot([], [], [], 'o', color='magenta', markersize=12, label='Drone 2')
    trail2, = ax.plot([], [], [], '-', color='magenta', alpha=0.3, linewidth=1)

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    height_text = ax.text2D(0.05, 0.88, '', transform=ax.transAxes)
    section_text = ax.text2D(0.05, 0.81, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    # Safety planes
    xx, yy = np.meshgrid([-2.0, 2.0], [-2.0, 2.0])
    ax.plot_surface(xx, yy, np.full_like(xx, 0.5), alpha=0.05, color='red')
    ax.plot_surface(xx, yy, np.full_like(xx, 2.5), alpha=0.05, color='blue')
    ax.plot_surface(xx, yy, np.full_like(xx, 1.5), alpha=0.03, color='yellow')

    start_wall_time = [None]

    def init():
        pygame.mixer.music.play()
        start_wall_time[0] = time.time()
        drone1.set_data([], [])
        drone1.set_3d_properties([])
        trail1.set_data([], [])
        trail1.set_3d_properties([])
        drone2.set_data([], [])
        drone2.set_3d_properties([])
        trail2.set_data([], [])
        trail2.set_3d_properties([])
        return drone1, trail1, drone2, trail2, time_text, height_text, section_text

    def update(i):
        if start_wall_time[0] is None:
            return drone1, trail1, drone2, trail2, time_text, height_text, section_text

        t = time.time() - start_wall_time[0]

        if t >= duration:
            pygame.mixer.music.stop()
            plt.close()
            return drone1, trail1, drone2, trail2, time_text, height_text, section_text

        frame = min(int(t * fps), total_frames - 1)

        # Drone 1
        drone1.set_data([x1[frame]], [y1[frame]])
        drone1.set_3d_properties([z1[frame]])
        start = max(0, frame - trail_length)
        trail1.set_data(x1[start:frame], y1[start:frame])
        trail1.set_3d_properties(z1[start:frame])

        # Drone 2
        if frame >= drone2_start:
            drone2.set_data([x2[frame]], [y2[frame]])
            drone2.set_3d_properties([z2[frame]])
            start2 = max(drone2_start, frame - trail_length)
            trail2.set_data(x2[start2:frame], y2[start2:frame])
            trail2.set_3d_properties(z2[start2:frame])
        else:
            drone2.set_data([], [])
            drone2.set_3d_properties([])
            trail2.set_data([], [])
            trail2.set_3d_properties([])

        if t < 55.5:
            section = "Section 1 — Drone 1 only"
        else:
            section = "Section 2 — Both Drones"

        time_text.set_text(f'Time: {t:.1f}s / {duration:.0f}s')
        height_text.set_text(f'D1: {z1[frame]:.2f}m | D2: {z2[frame]:.2f}m')
        section_text.set_text(section)

        return drone1, trail1, drone2, trail2, time_text, height_text, section_text

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