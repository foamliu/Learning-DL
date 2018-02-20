import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

file = '/Users/yangliu/code/virtualads/web/video/audio.mp3'
#y, sr = librosa.load(librosa.util.example_audio_file())
y, sr = librosa.load(file)
librosa.output.write_wav('file_trim_5s.wav', y, sr)
print(sr)
print(len(y))

plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr)
plt.title('Monophonic')


plt.figure(figsize=(12, 8))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

plt.subplot(4, 2, 2)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

plt.show()