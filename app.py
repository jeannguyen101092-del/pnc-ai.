import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, requests
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        img = img.crop((w*0.20, h*0.12, w*0.80, h*0.50))
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.5).convert('RGB')

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            norm = np.linalg.norm(vec)
            return (vec / norm).astype(float).tolist() if norm > 0 else vec.tolist()
    except:
        return None

# ================= SCRAPER =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None

    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        doc.close()

        return {"all_specs": {}, "img": img_bytes}
    except:
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.title("PPJ GROUP")

    try:
        count = supabase.table("ai_data").select("id", count="exact").execute().count or 0
    except:
        count = 0

    st.metric("Models", count)

# ================= MAIN =================
st.title("AI AUDITOR")

mode = st.radio("Mode", ["Audit Mode", "Version Control"])

# ================= AUDIT =================
if mode == "Audit Mode":

    f_audit = st.file_uploader("Upload PDF", type="pdf", key=f"up_{st.session_state['up_key']}")

    if f_audit:
        target = extract_full_data(f_audit.getvalue())

        if target and target.get('img'):

            st.image(target['img'], width=300)

            t_name = f_audit.name.upper()

            # ===== FIX PHÂN LOẠI =====
            def detect_type(name):
                if any(x in name for x in ["PANT","TROUSER","JEAN","LONG"]):
                    return "BOTTOM_LONG"
                if any(x in name for x in ["SHORT","SHRT","1/2"]):
                    return "BOTTOM_SHORT"
                if any(x in name for x in ["SHIRT","TEE","JACKET","TOP"]):
                    return "TOP"
                return "UNKNOWN"

            t_type = detect_type(t_name)

            res = supabase.table("ai_data").select("id, vector, file_name, image_url").execute()

            if res.data:
                t_vec = np.array(get_vector(target['img'])).reshape(1, -1)

                valid = []

                for r in res.data:
                    if not r['vector']: continue

                    r_name = r['file_name'].upper()
                    r_type = detect_type(r_name)

                    # ===== HARD FILTER (FIX CHÍNH) =====
                    if t_type != "UNKNOWN" and r_type != t_type:
                        continue

                    sim = cosine_similarity(
                        t_vec,
                        np.array(r['vector']).reshape(1,-1)
                    ).flatten()[0]

                    # BONUS
                    if t_type == r_type:
                        sim += 0.1

                    r['sim'] = sim
                    valid.append(r)

                if not valid:
                    st.error("❌ Không có mẫu cùng loại")
                else:
                    df = pd.DataFrame(valid).sort_values("sim", ascending=False).head(8)

                    st.subheader(f"Top match: {t_type}")

                    for i in range(0, len(df), 4):
                        cols = st.columns(4)
                        for j in range(4):
                            if i+j < len(df):
                                item = df.iloc[i+j]
                                with cols[j]:
                                    st.image(item['image_url'])
                                    st.write(f"{item['sim']:.2%}")
                                    if st.button("CHỌN", key=f"b{i+j}"):
                                        st.session_state['sel_audit'] = item

# ================= VERSION =================
if mode == "Version Control":

    if st.button("RESET"):
        st.session_state['up_key'] += 1
        st.session_state['ver_results'] = None
        st.rerun()

    f1 = st.file_uploader("File A", key="a")
    f2 = st.file_uploader("File B", key="b")

    if f1 and f2:
        st.success("Loaded 2 files")
