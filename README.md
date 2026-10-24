# Pill & Tablet Inspector

Aplikasi web berbasis **Python, Streamlit, dan YOLO (Ultralytics)** untuk mendeteksi dan menghitung objek pil dan tablet berdasarkan gambar.

## Fitur

- Mengunggah gambar berformat PNG, JPG, JPEG, WEBP, atau BMP.
- Mendeteksi objek menggunakan model YOLO dari file `best.pt`.
- Menampilkan kotak deteksi, label kelas, dan nilai confidence.
- Menghitung jumlah objek yang terdeteksi pada setiap kelas.
- Mengatur confidence threshold dan ukuran gambar inferensi.
- Menampilkan waktu pemrosesan dan rata-rata confidence.
- Mengunduh gambar hasil inspeksi.

## Kelas Deteksi

| Nama Kelas Model | Label Aplikasi |
|---|---|
| `damaged-pill` | Pil Rusak |
| `foreign-matter` | Benda Asing |
| `intact-pill` | Pil Utuh |
| `intact-tablet` | Tablet Utuh |

## Struktur Folder

pill_tablet_inspector/
├── app.py
├── best.pt
├── requirements.txt
└── README.md

File `best.pt` merupakan model YOLO hasil training dan harus berada di folder yang sama dengan `app.py`.

## Persyaratan

- Python 3.10–3.12 disarankan.
- Windows, macOS, atau Linux.
- Koneksi internet untuk instalasi library jika diperlukan.

## Instalasi di Windows

Buka PowerShell di folder project, lalu jalankan perintah berikut.

### 1. Membuat Virtual Environment

py -3.12 -m venv .venv

### 2. Menginstal Library

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

### 3. Menjalankan Aplikasi

.\.venv\Scripts\python.exe -m streamlit run app.py

Setelah aplikasi berjalan, buka browser dan akses:

`http://localhost:8501`

## Mengakses dari Perangkat Lain

Untuk mengakses aplikasi dari perangkat lain yang terhubung ke Wi-Fi yang sama, jalankan:

.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0

Cari alamat IPv4 laptop menggunakan:

ipconfig

Kemudian, pada perangkat lain, buka:

`http://ALAMAT-IP-LAPTOP:8501`

Ganti `ALAMAT-IP-LAPTOP` dengan alamat IPv4 laptop yang menjalankan aplikasi. Pastikan Windows Firewall mengizinkan koneksi jika diperlukan.

## Catatan Model

- Aplikasi menjalankan inferensi menggunakan model `best.pt`; aplikasi tidak melatih model secara langsung.
- Label Bahasa Indonesia hanya mengubah tampilan nama kelas, bukan kelas yang tersimpan dalam model.
- Jika model terus memprediksi satu kelas, periksa dataset, anotasi, hasil evaluasi, dan proses training.
- Confidence threshold yang lebih tinggi dapat menyaring prediksi dengan keyakinan rendah, tetapi juga berisiko melewatkan objek.
- Hasil deteksi merupakan prediksi AI dan bukan jaminan keamanan atau kelayakan obat untuk dikonsumsi.