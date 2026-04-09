import librosa
import numpy as np

def beat_analysis(audio_path):
    y, sr = librosa.load(audio_path)

    # Perform beat and tempo analysis 
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'onset_times': onset_times,
        'rms': rms,
        'rms_times': rms_times,
        'duration': librosa.get_duration(y=y, sr=sr)
    }

if __name__ == "__main__":
    data = beat_analysis('../audio/robots_mixdown.mp3')
    print(f"Tempo: {data['tempo']} BPM")
    print(f"Total beats: {len(data['beat_times'])}")
    print(f"Total onsets: {len(data['onset_times'])}")
    print(f"Duration: {data['duration']:.1f} seconds")
    print(f"First 5 beat times: {data['beat_times'][:5]}")
