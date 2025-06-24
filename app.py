import streamlit as st
import pandas as pd
import joblib

# Muat model yang sudah dilatih
try:
    model = joblib.load('model_graduation.pkl')
except FileNotFoundError:
    st.error("File 'model_graduation.pkl' tidak ditemukan. Pastikan model berada di direktori yang sama dengan aplikasi Streamlit Anda.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan

# Judul dan Deskripsi Aplikasi
st.title('Prediksi Kategori Waktu Lulus Mahasiswa')
st.write("""
Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus **Tepat Waktu** atau **Terlambat**
berdasarkan beberapa faktor akademik dan latar belakang keluarga.
""")

# --- Bagian Input Data Pengguna (Sidebar) ---
st.sidebar.header('Input Data Mahasiswa')

def user_input_features():
    act = st.sidebar.slider('ACT Composite Score', 1, 36, 25)
    sat = st.sidebar.slider('SAT Total Score', 400, 1600, 1200)
    gpa = st.sidebar.slider('High School GPA', 1.0, 4.0, 3.5)
    income = st.sidebar.slider('Pendapatan Orang Tua ($)', 0, 200000, 75000)
    education = st.sidebar.slider('Tingkat Pendidikan Orang Tua (Numerik)', 0, 20, 12) # Sesuaikan skala jika perlu

    data = {
        'ACT composite score': act,
        'SAT total score': sat,
        'high school gpa': gpa,
        'parental income': income,
        'parent_edu_numerical': education
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Ambil input dari pengguna
df_input = user_input_features()

# --- Tampilkan Input Pengguna di Halaman Utama ---
st.subheader('Data Input yang Anda Masukkan')
st.write(df_input)

# --- Tombol untuk Melakukan Prediksi ---
if st.button('Lakukan Prediksi Kelulusan'):
    # Lakukan prediksi menggunakan model yang dimuat
    predicted_code = model.predict(df_input)[0]

    # Konversi hasil prediksi (0 atau 1) ke label yang mudah dibaca
    label_mapping = {1: 'Tepat Waktu', 0: 'Terlambat'}
    predicted_label = label_mapping.get(predicted_code, 'Tidak Diketahui')

    # --- Tampilkan Hasil Prediksi ---
    st.subheader('Hasil Prediksi')
    if predicted_label == 'Tepat Waktu':
        st.success(f'Berdasarkan data ini, mahasiswa diprediksi akan lulus: **{predicted_label}** 🎉')
    else:
        st.warning(f'Berdasarkan data ini, mahasiswa diprediksi akan lulus: **{predicted_label}** ⏳')

st.markdown("""
---
*Catatan: Pastikan skala untuk 'Tingkat Pendidikan Orang Tua (Numerik)' di aplikasi ini konsisten dengan skala yang digunakan saat Anda melatih model machine learning Anda.*
""")