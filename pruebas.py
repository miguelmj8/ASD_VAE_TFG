import common as com
import matplotlib.pyplot as plt

audio_dir = "../data/data/dev_valve/valve/test/section_00_source_test_anomaly_0000_pat_00.wav"
y, sr = com.load_audio(audio_dir)
NFFT = 1024
n_mels = 128

com.play_audio(y, sr)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 4))
com.plot_audio(y, sr, ax=ax1, xlim=(0,5),title="Audio Signal")
com.plot_mag_melspectrogram(y, sr, NFFT, NFFT//2, n_mels = n_mels, ax=ax2,title="Mel-Spectrogram")
plt.show(block=True)
