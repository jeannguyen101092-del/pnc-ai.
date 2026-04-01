# ==========================================================
# BẢN V78: FIX CHUẨN LOGIC LẤY SPEC - GIỮ NGUYÊN CẤU TRÚC
# ==========================================================

import streamlit as st
import os, fitz, io, pickle, torch, pdfplumber, re, zlib, pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="AI SMART SPEC V78", page_icon="🚀")

folder_path = '/content/drive/MyDrive/PNC_PDF'
db_path = os.path.join(folder_path, 'db_master_v39.pkl')

# ================= LOGIN GIỮ NGUYÊN =================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 ĐĂNG NHẬP HỆ THỐNG AI")
    user = st.text_input("Tài khoản admin:")
    pw = st.text_input("Mật khẩu:", type="password")
    if st.button("Đăng nhập"):
        if user == "admin" and pw == "123456":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Sai tài khoản hoặc mật khẩu!")

if not st.session_state.logged_in:
    login()
    st.stop()

# ================= DB =================
def save_db(data_dict):
    with open(db_path, "wb") as f:
        f.write(zlib.compress(pickle.dumps(data_dict)))

if 'db' not in st.session_state:
    if os.path.exists(db_path):
        try:
            with open(db_path, "rb") as f:
                st.session_state.db = pickle.loads(zlib.decompress(f.read()))
        except:
            st.session_state.db = {}
    else:
        st.session_state.db = {}

# ================= AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ================= FIX CORE =================
def clean_pom_name(text):
    if not text:
        return ""
    t = str(text).upper().strip()
    t = re.sub(r'^\d+[\.\-\)]\s*', '', t)
    return t


def parse_val(t):
    if t is None:
        return None
    t_str = str(t).replace(',', '.')
    found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
    if not found:
        return None
    v = found[-1]  # FIX: lấy số cuối
    try:
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except:
        return None

# ================= CHỌN SIZE =================
selected_size = st.sidebar.selectbox("📏 Chọn size chuẩn", ["S", "M", "L", "XL", "XXL", "30", "31", "32", "33", "34"])

# ================= PARSE PDF FIX =================
def get_data(pdf_path):
    specs = {}
    sample_sz = selected_size
    all_texts = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    all_texts.append(txt)

                for table in p.extract_tables():
                    if not table or len(table) < 2:
                        continue

                    target_idx = -1

                    # FIX: tìm đúng size user chọn
                    for r_idx in range(min(5, len(table))):
                        for c_idx, val in enumerate(table[r_idx]):
                            if str(val).strip().upper() == selected_size:
                                target_idx = c_idx
                                break
                        if target_idx != -1:
                            break

                    if target_idx == -1:
                        continue

                    for r in table:
                        if len(r) <= target_idx:
                            continue

                        # FIX: lấy POM đúng cột
                        raw_pom = str(r[0]) if r[0] else ""
                        pom_n = clean_pom_name(raw_pom)

                        val = parse_val(r[target_idx])

                        if val is not None and len(pom_n) > 2:
                            if pom_n in specs:
                                pom_n = pom_n + "_2"
                            specs[pom_n] = val

        # ===== IMAGE =====
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('RGB')
        img.thumbnail((500, 500))
        buf = io.BytesIO()
        img.save(buf, format="WEBP")

        return {
            "spec": specs,
            "img_bytes": buf.getvalue(),
            "sample": sample_sz,
        }

    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# ================= UI =================
st.title("👖 AI SMART SPEC V78")

up = st.file_uploader("📥 Upload Spec PDF", type="pdf")

if up:
    with open("temp.pdf", "wb") as f:
        f.write(up.getbuffer())

    data = get_data("temp.pdf")

    if data:
        st.success("✅ Đọc spec thành công")

        df = pd.DataFrame(list(data['spec'].items()), columns=["POM", "VALUE"])
        st.dataframe(df, use_container_width=True)

        st.write(f"📏 Size đang dùng: {selected_size}")

# ================= END =================
