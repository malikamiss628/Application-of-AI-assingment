import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import soundfile as sf

# ==============================
# 1. LOAD AUDIO
# ==============================

clean_path = "clean.wav"   # your clean speech
noise_path = "noise.wav"   # blower noise

clean, sr = librosa.load(clean_path, sr=16000)
noise, _ = librosa.load(noise_path, sr=16000)

# Make same length
min_len = min(len(clean), len(noise))
clean = clean[:min_len]
noise = noise[:min_len]

# Create noisy signal
noisy = clean + 0.5 * noise

# ==============================
# 2. CONVERT TO SPECTROGRAM
# ==============================

def to_spec(signal):
    return np.abs(librosa.stft(signal, n_fft=512, hop_length=256))

clean_spec = to_spec(clean)
noisy_spec = to_spec(noisy)

# Normalize
clean_spec = clean_spec / np.max(clean_spec)
noisy_spec = noisy_spec / np.max(noisy_spec)

# Transpose (time steps as samples)
X = noisy_spec.T
Y = clean_spec.T

# ==============================
# 3. BUILD AUTOENCODER
# ==============================

model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(X.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

model.summary()

# ==============================
# 4. TRAIN MODEL
# ==============================

history = model.fit(X, Y,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.1)

# ==============================
# 5. PLOT EPOCH vs ERROR
# ==============================

plt.figure()
plt.plot(history.history['loss'], label='Train Error')
plt.plot(history.history['val_loss'], label='Validation Error')
plt.xlabel("Epochs")
plt.ylabel("Error (MSE)")
plt.title("Epoch vs Error")
plt.legend()
plt.show()

# ==============================
# 6. PREDICT CLEAN SPEECH
# ==============================

pred_spec = model.predict(X)

# Transpose back
pred_spec = pred_spec.T

# ==============================
# 7. RECONSTRUCT AUDIO
# ==============================

# Use original phase
phase = np.angle(librosa.stft(noisy, n_fft=512, hop_length=256))

reconstructed = pred_spec * np.exp(1j * phase)

clean_audio = librosa.istft(reconstructed, hop_length=256)

# Save output
sf.write("denoised_output.wav", clean_audio, sr)

# ==============================
# 8. ACCURACY CALCULATION
# ==============================

mse = np.mean((Y - model.predict(X))**2)
accuracy = 1 - mse

print("Final MSE:", mse)
print("Accuracy:", accuracy)