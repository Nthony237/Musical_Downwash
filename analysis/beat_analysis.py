import librosa
import numpy as np

def extract_features(filepath):
    """Extract key features for choreography mapping."""
    y, sr = librosa.load(filepath)

    # Tempo & Rhythm
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=75, tightness=100)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Dynamics
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Frequency bands
    harmonic, percussive = librosa.effects.hpss(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    bass_energy = np.mean(mel_db[:20, :], axis=0)
    treble_energy = np.mean(mel_db[80:, :], axis=0)
    freq_times = librosa.frames_to_time(np.arange(len(bass_energy)), sr=sr)

    # Structure (3 equal parts)
    duration = librosa.get_duration(y=y, sr=sr)
    third = duration / 3
    section_times = [0, third, third*2, duration]

    # Onset strength for beat drops - lowered threshold to 0.65
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_strength_norm = onset_strength / onset_strength.max()
    drop_frames = np.where(onset_strength_norm > 0.65)[0]
    drop_times = librosa.frames_to_time(drop_frames, sr=sr)

    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'rms': rms,
        'rms_times': rms_times,
        'bass_energy': bass_energy,
        'treble_energy': treble_energy,
        'freq_times': freq_times,
        'section_times': section_times,
        'duration': duration,
        'sr': sr,
        'drop_times': drop_times
    }

if __name__ == "__main__":
    filepath = '../audio/robots_mixdown.mp3'
    features = extract_features(filepath)
    print(f"Tempo: {features['tempo']} BPM")
    print(f"Duration: {features['duration']:.1f}s")
    print(f"Beats: {len(features['beat_times'])}")
    print(f"Drop times: {len(features['drop_times'])} detected")
    print(f"\n--- Section Splits ---")
    labels = ['Intro (Drone 1)', 'Build (Drone 1+2)', 'Climax (All Drones)']
    for i, (t, label) in enumerate(zip(features['section_times'], labels)):
        print(f"Section {i+1} - {label}: starts at {t:.1f}s")