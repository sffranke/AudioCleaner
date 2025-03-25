import streamlit as st
import numpy as np
import io
import pydub
from df.enhance import enhance, init_df
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import tempfile

st.title("🚀 Enhanced Speech with DeepFilterNet3")
st.subheader("High-quality speech enhancement powered by DeepFilterNet3\n\nfmpeg -i input.mp3 -ac 1 -ar 48000 -f segment -segment_time 650 -segment_format wav -fs 100M output_%03d.wav")

uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "m4a", "wav", "flac", "ogg"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio = pydub.AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()))
        audio.export(temp_audio_file.name, format="wav")
        temp_audio_file.seek(0)
        raw_audio = temp_audio_file.read()

    st.audio(raw_audio, format='audio/wav')

    if st.button('✨ Enhance Audio'):
        audio_stream = io.BytesIO(raw_audio)
        waveform, sample_rate = torchaudio.load(audio_stream, backend='soundfile')

        # Resample auf 48 kHz falls nötig
        if sample_rate != 48000:
            resampler = T.Resample(sample_rate, 48000)
            waveform = resampler(waveform)
            sample_rate = 48000

        # Normalisiere den Pegel
        waveform = waveform / waveform.abs().max() * 0.8

        waveform_cpu = waveform.cpu()

        # Lade DeepFilterNet3 Modell
        model, df_state, _ = init_df(model_base_dir='DeepFilterNet3')
        model = model.to(device)

        # Audio verbessern
        enhanced = enhance(model, df_state, waveform_cpu)

        # Mischung auf CPU durchführen
        strength = 0.95
        enhanced = strength * enhanced + (1 - strength) * waveform_cpu

        enhanced_numpy = enhanced.numpy()

        st.success('✅ Audio enhanced successfully!')
        st.audio(enhanced_numpy, format='audio/wav', sample_rate=sample_rate)

        # Spektrogramme darstellen
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        orig_spec = librosa.feature.melspectrogram(y=waveform_cpu.squeeze().numpy(), sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(orig_spec, ref=np.max), sr=sample_rate,
                                 x_axis='time', y_axis='mel', ax=axs[0])
        axs[0].set(title='Original Audio')

        enhanced_spec = librosa.feature.melspectrogram(y=enhanced_numpy.squeeze(), sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(enhanced_spec, ref=np.max), sr=sample_rate,
                                 x_axis='time', y_axis='mel', ax=axs[1])
        axs[1].set(title='Enhanced Audio')

        st.pyplot(fig)

        # Enhanced Audio korrekt zum Download bereitstellen
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, enhanced_numpy.squeeze(), sample_rate, format='WAV')
        audio_buffer.seek(0)

        original_filename = uploaded_file.name.rsplit('.', 1)[0]
        download_filename = f"{original_filename}-besser.wav"

        st.download_button("⬇️ Download Enhanced Audio", audio_buffer, download_filename, mime="audio/wav")

        st.markdown("---")
        st.markdown("🔗 [DeepFilterNet3 GitHub](https://github.com/Rikorose/DeepFilterNet)")

        # GPU-Ressourcen explizit freigeben
        import gc

        del model, enhanced, waveform
        torch.cuda.empty_cache()
        gc.collect()
