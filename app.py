import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import io
import zipfile
import pandas as pd
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import maximum_filter

st.set_page_config(page_title="Song Identifier", layout="wide")

def fast_spectrogram(signal, sr, window_size=2048, hop_size=512):
    noverlap = window_size - hop_size
    freqs, times, Sxx = scipy_spectrogram(signal, fs=sr, window='hamming',
                                           nperseg=window_size, noverlap=noverlap,
                                           scaling='spectrum', mode='magnitude')
    spec_db = 20 * np.log10(Sxx + 1e-6)
    return spec_db, freqs, times


def find_peaks(spec_db, freqs, times, neighborhood_size=10, threshold=-40):
    local_max = maximum_filter(spec_db, size=neighborhood_size) == spec_db
    above_threshold = spec_db > threshold
    peak_mask = local_max & above_threshold
    freq_idx, time_idx = np.where(peak_mask)

    peak_list = []
    for f_i, t_i in zip(freq_idx, time_idx):
        peak_list.append((times[t_i], freqs[f_i]))
    return peak_list


def generate_hashes(peak_list, fan_out=5, max_time_gap=2.0):
    hashes = []
    peak_list = sorted(peak_list, key=lambda p: p[0])
    n = len(peak_list)
    for i in range(n):
        t1, f1 = peak_list[i]
        count = 0
        for j in range(i+1, n):
            t2, f2 = peak_list[j]
            dt = t2 - t1
            if dt <= 0:
                continue
            if dt > max_time_gap:
                break
            hash_key = (round(f1), round(f2), round(dt, 2))
            hashes.append((hash_key, t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


def fingerprint_song(filepath, window_size=2048, hop_size=512):
    y, sr = librosa.load(filepath, sr=22050, mono=True)
    spec_db, freqs, times = fast_spectrogram(y, sr, window_size, hop_size)
    peaks = find_peaks(spec_db, freqs, times)
    hashes = generate_hashes(peaks)
    return hashes


@st.cache_resource
def build_database(song_folder="songs"):
    database = {}
    song_names = {}

    song_files = [f for f in os.listdir(song_folder) if f.lower().endswith(".mp3")]

    for song_id, fname in enumerate(song_files):
        path = os.path.join(song_folder, fname)
        try:
            hashes = fingerprint_song(path)
        except Exception as e:
            continue

        song_names[song_id] = fname

        for hash_key, anchor_time in hashes:
            database.setdefault(hash_key, []).append((song_id, anchor_time))

    return database, song_names


def identify_clip(audio_path, database, song_names, clip_seconds=10):
    y_q, sr_q = librosa.load(audio_path, sr=22050, mono=True, duration=clip_seconds)
    spec_q, freqs_q, times_q = fast_spectrogram(y_q, sr_q, 2048, 512)
    peaks_q = find_peaks(spec_q, freqs_q, times_q)
    query_hashes = generate_hashes(peaks_q)

    offset_votes = {}
    for hash_key, q_time in query_hashes:
        if hash_key in database:
            for (song_id, db_time) in database[hash_key]:
                offset = round(db_time - q_time, 1)
                offset_votes.setdefault(song_id, {})
                offset_votes[song_id][offset] = offset_votes[song_id].get(offset, 0) + 1

    best_song, best_score, best_votes = None, 0, {}
    for song_id, votes in offset_votes.items():
        top = max(votes.values())
        if top > best_score:
            best_score, best_song, best_votes = top, song_id, votes

    return best_song, best_score, best_votes, spec_q, freqs_q, times_q, peaks_q


st.title("🎵 Song Identifier")

with st.spinner("Indexing song database... (only happens once)"):
    database, song_names = build_database("songs")

st.success(f"Database ready: {len(song_names)} songs indexed, {len(database)} unique hashes.")

mode = st.radio("Choose mode:", ["Single Clip", "Batch Mode"])

# MODE 1: SINGLE CLIP

if mode == "Single Clip":
    uploaded = st.file_uploader("Upload a query clip (mp3/wav)", type=["mp3", "wav"])

    if uploaded is not None:
        with open("temp_query.mp3", "wb") as f:
            f.write(uploaded.read())

        best_song, best_score, best_votes, spec_q, freqs_q, times_q, peaks_q = identify_clip(
            "temp_query.mp3", database, song_names
        )

        st.subheader("Step 1: Spectrogram")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.imshow(spec_q, aspect='auto', origin='lower',
                   extent=[times_q[0], times_q[-1], freqs_q[0], freqs_q[-1]])
        ax1.set_ylim(0, 5000)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (Hz)")
        st.pyplot(fig1)

        st.subheader("Step 2: Constellation of Peaks")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.imshow(spec_q, aspect='auto', origin='lower', cmap='gray',
                   extent=[times_q[0], times_q[-1], freqs_q[0], freqs_q[-1]])
        ax2.scatter([p[0] for p in peaks_q], [p[1] for p in peaks_q], color='red', s=8)
        ax2.set_ylim(0, 5000)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        st.pyplot(fig2)

        st.subheader("Step 3: Offset Histogram (best match)")
        if best_votes:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.bar(best_votes.keys(), best_votes.values())
            ax3.set_xlabel("Offset (s)")
            ax3.set_ylabel("Matching hashes")
            st.pyplot(fig3)
        else:
            st.write("No matching hashes found at all.")

        st.subheader("Result")
        if best_song is not None and best_score >= 5:
            matched_name = os.path.splitext(song_names[best_song])[0]
            st.success(f"**Matched Song:** {matched_name}")
            st.write(f"Confidence (votes at best offset): {best_score}")
        else:
            st.error("No confident match found.")

# MODE 2: BATCH MODE

else:
    st.write("Upload multiple query clips. The app will identify each one and "
             "let you download a `results.csv` with two columns: filename, prediction.")

    uploaded_files = st.file_uploader(
        "Upload query clips (mp3/wav)", type=["mp3", "wav"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Run Batch Identification"):
        results = []
        progress = st.progress(0)

        for i, uploaded in enumerate(uploaded_files):
            temp_path = f"temp_batch_{i}.mp3"
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            best_song, best_score, _, _, _, _, _ = identify_clip(temp_path, database, song_names)

            if best_song is not None:
                prediction = os.path.splitext(song_names[best_song])[0]
            else:
                prediction = "NO_MATCH"

            results.append({"filename": uploaded.name, "prediction": prediction})
            progress.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(results, columns=["filename", "prediction"])
        st.dataframe(df)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv"
        )