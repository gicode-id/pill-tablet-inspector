# ============================================================
# 1. IMPORT LIBRARY
# Library untuk file, penghitungan, gambar, waktu, UI Streamlit,
# pemrosesan gambar, dan model YOLO.
from pathlib import Path
from collections import Counter
from io import BytesIO
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# 2. LOKASI FILE DAN KONFIGURASI KELAS
# APP_DIR adalah folder app.py. File best.pt diharapkan berada di sini.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"

# Nama kelas yang digunakan model training -> label yang ditampilkan di aplikasi.
CLASS_LABELS_ID = {
    "damaged-pill": "Pil Rusak",
    "foreign-matter": "Benda Asing",
    "intact-pill": "Pil Utuh",
    "intact-tablet": "Tablet Utuh",
}
CLASS_ORDER = ["Pil Rusak", "Benda Asing", "Pil Utuh", "Tablet Utuh"]
CLASS_COLORS = {
    "Pil Rusak": "#fb7185",
    "Benda Asing": "#fbbf24",
    "Pil Utuh": "#34d399",
    "Tablet Utuh": "#60a5fa",
}
CLASS_COLOR_RGB = {
    "Pil Rusak": (251, 113, 133),
    "Benda Asing": (251, 191, 36),
    "Pil Utuh": (52, 211, 153),
    "Tablet Utuh": (96, 165, 250),
}

# ============================================================
# 3. PEMETAAN NAMA KELAS
# Menyamakan format nama kelas dan menerjemahkan label model ke Indonesia.
def normalize_class_name(name):
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")

def translate_class_name(name):
    normalized = normalize_class_name(name)
    return CLASS_LABELS_ID.get(normalized, str(name))

# ============================================================
# 4. MEMUAT MODEL YOLO
# Cache mencegah model dibaca ulang dari disk pada setiap interaksi.
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ============================================================
# 5. KARTU METRIK
# Membuat kartu ringkasan seperti total deteksi dan confidence.
def metric(label, value, foot, color=None):
    border = f"border-top:3px solid {color};" if color else ""
    st.markdown(
        f"""<div class="glass-card" style="{border}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-foot">{foot}</div></div>""",
        unsafe_allow_html=True,
    )

# ============================================================
# 6. GAMBAR HASIL DETEKSI
# Menggambar kotak, label Indonesia, dan nilai confidence pada gambar.
def draw_translated_boxes(image_rgb, result, names, show_boxes=True, show_labels=True):
    """Gambar kotak sendiri agar label di gambar memakai Bahasa Indonesia."""
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    if result.boxes is None:
        return np.array(image)

    for box in result.boxes:
        cls_id = int(box.cls.item())
        raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        label = translate_class_name(raw_name)
        confidence_value = float(box.conf.item())
        color = CLASS_COLOR_RGB.get(label, (103, 232, 249))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if show_boxes:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        if show_labels:
            text = f"{label} {confidence_value:.0%}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = max(0, x1)
            ty = max(0, y1 - text_h - 8)
            draw.rectangle((tx, ty, tx + text_w + 10, ty + text_h + 8), fill=color)
            draw.text((tx + 5, ty + 3), text, fill=(11, 16, 32), font=font)

    return np.array(image)

# ============================================================
# 7. PENGATURAN HALAMAN STREAMLIT
# Mengatur judul tab, ikon, layout, dan kondisi awal sidebar.
st.set_page_config(
    page_title="Pill & Tablet Inspector | AI Vision",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 8. CSS / DESAIN ANTARMUKA
# Mengatur warna, font, kartu, tombol, jarak, dan gaya halaman.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp {
  background: radial-gradient(circle at 82% 0%, rgba(124,58,237,.13), transparent 28rem),
              radial-gradient(circle at 10% 8%, rgba(6,182,212,.10), transparent 24rem), #0b1020;
  color: #e5e7eb; font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(11,16,32,.72); }
[data-testid="stSidebar"] { background: #0d1424; border-right: 1px solid rgba(148,163,184,.18); }
[data-testid="stSidebar"] * { color: #e5e7eb; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.03em; }
.eyebrow { color:#67e8f9; font-size:.76rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3.4rem); font-weight:700; line-height:1.05; color:#f8fafc; }
.hero-copy { color:#a8b5ca; font-size:1.02rem; max-width:760px; line-height:1.65; margin-top:.8rem; }
.glass-card { background:linear-gradient(145deg,rgba(23,34,56,.94),rgba(15,23,42,.94)); border:1px solid rgba(148,163,184,.18); border-radius:18px; padding:1.1rem 1.2rem; box-shadow:0 18px 55px rgba(0,0,0,.12); height:100%; }
.metric-label { color:#9aabc3; font-size:.83rem; margin-bottom:.45rem; }
.metric-value { font-family:'Space Grotesk',sans-serif; color:#f8fafc; font-size:2.2rem; font-weight:700; line-height:1.1; }
.metric-foot { color:#7dd3fc; font-size:.78rem; margin-top:.55rem; }
.section-title { color:#f8fafc; font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:600; margin:.4rem 0 .8rem; }
.muted { color:#94a3b8; }
.status-pill { display:inline-block; padding:.35rem .65rem; border-radius:999px; color:#a7f3d0; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.25); font-size:.76rem; font-weight:600; }
.tiny-note { color:#8190a8; font-size:.78rem; }
div[data-testid="stFileUploader"] { background:rgba(15,23,42,.55); border:1px dashed rgba(103,232,249,.35); padding:1rem; border-radius:14px; }
div.stButton>button,div.stDownloadButton>button { border-radius:11px; border:1px solid rgba(103,232,249,.35); background:linear-gradient(100deg,#0891b2,#6366f1); color:white; font-weight:700; min-height:2.8rem; }
div.stButton>button:hover,div.stDownloadButton>button:hover { border-color:rgba(103,232,249,.7); color:white; filter:brightness(1.08); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 9. SIDEBAR PENGATURAN
# Mengatur confidence, ukuran gambar, label/kotak, dan memeriksa best.pt.
with st.sidebar:
    st.markdown("## 💊 Pill & Tablet Inspector")
    st.caption("AI Vision · Quality inspection")
    st.markdown("---")
    st.markdown("### Detection settings")
    confidence = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Naikkan untuk menyaring deteksi yang kurang yakin. Terlalu tinggi bisa melewatkan objek.",
    )
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
    show_labels = st.toggle("Tampilkan label & confidence", value=True)
    show_boxes = st.toggle("Tampilkan kotak deteksi", value=True)
    st.markdown("---")
    st.markdown("**File model**")
    if MODEL_PATH.exists():
        st.markdown('<span class="status-pill">● best.pt ditemukan</span>', unsafe_allow_html=True)
        st.caption(str(MODEL_PATH))
    else:
        st.error("File `best.pt` belum ditemukan.")
        st.caption("Letakkan best.pt di folder yang sama dengan app.py.")
    st.markdown("---")
    st.caption("Gambar diproses secara lokal di laptop.")

# ============================================================
# 10. HEADER DAN INPUT GAMBAR
# Menampilkan judul aplikasi dan menyediakan uploader gambar.
st.markdown('<div class="eyebrow">COMPUTER VISION WORKSPACE / QUALITY CONTROL</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Pill & Tablet<br>Quality Inspector.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Unggah foto untuk mendeteksi empat kelas model: pil rusak, benda asing, pil utuh, dan tablet utuh.</div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown('<div class="section-title">01 &nbsp; Input gambar</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih foto pil atau tablet", type=["png", "jpg", "jpeg", "webp", "bmp"])
    pil_image = None
    if uploaded_file:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption=f"Input · {uploaded_file.name}", use_container_width=True)
        except Exception as exc:
            st.error(f"Gambar tidak bisa dibuka: {exc}")
    else:
        st.markdown(
            '<div class="glass-card"><div class="muted">Belum ada gambar yang dipilih.</div>'
            '<p style="color:#e2e8f0;font-size:1.05rem;font-weight:600;">Mulai dengan mengunggah foto</p>'
            '<p class="tiny-note">Format PNG, JPG, JPEG, WEBP, atau BMP.</p></div>',
            unsafe_allow_html=True,
        )

# Ringkasan model dan tombol untuk memulai deteksi.
with right:
    st.markdown('<div class="section-title">02 &nbsp; Ringkasan model</div>', unsafe_allow_html=True)
    st.markdown("""<div class="glass-card"><div class="metric-label">Mesin deteksi</div>
    <div style="font-size:1.35rem;font-weight:700;color:#f8fafc;">YOLO · Ultralytics</div>
    <p class="tiny-note">Model: <code>best.pt</code></p><hr style="border-color:rgba(148,163,184,.16);margin:1rem 0;">
    <div class="metric-label">Empat kelas model</div>
    <p style="color:#cbd5e1;line-height:1.9;margin-bottom:.5rem;">🔴 Pil Rusak<br>🟡 Benda Asing<br>🟢 Pil Utuh<br>🔵 Tablet Utuh</p>
    <span class="status-pill">Deteksi lokal</span></div>""", unsafe_allow_html=True)
    st.write("")
    run_clicked = st.button(
        "Jalankan inspeksi", type="primary", use_container_width=True,
        disabled=(pil_image is None or not MODEL_PATH.exists()),
    )

# ============================================================
# 11. PROSES DETEKSI YOLO
# Menjalankan prediksi, menghitung kelas/confidence, dan menyimpan hasil.
if run_clicked and pil_image is not None:
    try:
        with st.spinner("YOLO sedang memeriksa gambar..."):
            model = load_model(str(MODEL_PATH))
            rgb = np.array(pil_image)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            started = time.perf_counter()
            result = model.predict(source=bgr, conf=confidence, imgsz=image_size, verbose=False)[0]
            elapsed_ms = (time.perf_counter() - started) * 1000
            names = result.names
            class_counts = Counter()
            confidences = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    class_counts[translate_class_name(raw_name)] += 1
                    confidences.append(float(box.conf.item()))
            annotated = draw_translated_boxes(rgb, result, names, show_boxes, show_labels)
            st.session_state["latest_result"] = {
                "image": annotated,
                "counts": dict(class_counts),
                "total": sum(class_counts.values()),
                "elapsed_ms": elapsed_ms,
                "confidences": confidences,
                "filename": uploaded_file.name if uploaded_file else "image",
                "model_names": list(names.values()) if isinstance(names, dict) else list(names),
            }
    except Exception as exc:
        st.error(f"Deteksi gagal: {exc}")
        st.exception(exc)

# ============================================================
# 12. MENAMPILKAN HASIL INSPEKSI
# Menampilkan metrik, hitungan empat kelas, gambar anotasi, dan unduhan.
latest = st.session_state.get("latest_result")
if latest:
    st.write("")
    st.markdown('<div class="section-title">03 &nbsp; Hasil inspeksi</div>', unsafe_allow_html=True)

    raw_normalized = {normalize_class_name(n) for n in latest["model_names"]}
    expected_model_classes = set(CLASS_LABELS_ID.keys())
    if not expected_model_classes.issubset(raw_normalized):
        st.warning(
            "Nama kelas yang tersimpan di best.pt tidak sama dengan empat kelas yang diharapkan. "
            f"Nama kelas model: {', '.join(latest['model_names'])}. "
            "Label yang tidak dikenal akan ditampilkan sesuai nama aslinya."
        )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric("Total terdeteksi", latest["total"], "Semua objek yang terdeteksi")
    with c2:
        metric("Waktu deteksi", f'{latest["elapsed_ms"]:.0f} ms', "Waktu pemrosesan gambar")
    with c3:
        avg = sum(latest["confidences"]) / len(latest["confidences"]) if latest["confidences"] else 0
        metric("Rata-rata confidence", f"{avg:.0%}" if latest["confidences"] else "—", "Keyakinan prediksi model")

    st.write("")
    class_cols = st.columns(4, gap="medium")
    for idx, class_name in enumerate(CLASS_ORDER):
        with class_cols[idx]:
            metric(class_name, latest["counts"].get(class_name, 0), "objek terdeteksi", CLASS_COLORS[class_name])

    c_left, c_right = st.columns([1.6, 1], gap="large")
    with c_left:
        st.markdown('<div class="section-title">Gambar hasil deteksi</div>', unsafe_allow_html=True)
        st.image(latest["image"], use_container_width=True)
        buffer = BytesIO()
        Image.fromarray(latest["image"]).save(buffer, format="PNG")
        st.download_button(
            "Unduh gambar hasil inspeksi",
            data=buffer.getvalue(),
            file_name="pill_tablet_inspection.png",
            mime="image/png",
            use_container_width=True,
        )
    with c_right:
        st.markdown('<div class="section-title">Rincian deteksi</div>', unsafe_allow_html=True)
        for class_name in CLASS_ORDER:
            st.write(f"**{class_name}: {latest['counts'].get(class_name, 0)}**")
        unknown = {
            name: count for name, count in latest["counts"].items()
            if name not in CLASS_ORDER
        }
        for class_name, count in unknown.items():
            st.write(f"**{class_name}: {count}**")
        if not latest["counts"]:
            st.info("Tidak ada objek terdeteksi. Coba turunkan confidence threshold.")
        st.caption(f'Sumber gambar: {latest["filename"]}')

# ============================================================
# 13. CATATAN BATASAN SISTEM
# Prediksi AI bukan jaminan keamanan atau kelayakan obat.
st.write("")
st.markdown(
    '<p class="tiny-note">Catatan: aplikasi menampilkan prediksi model, bukan memastikan keamanan obat untuk dikonsumsi. '
    'Jika semua objek diprediksi sebagai satu kelas, performa perlu diperiksa melalui dataset dan hasil training YOLO.</p>',
    unsafe_allow_html=True,
)
