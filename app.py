import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V46.2", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except:
        return None

model_ai = load_ai()

# ================= PARSE =================
def parse_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ['nan', '-', 'none', 'tol']): return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except:
        return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT =================
def extract_pom_deep_v462(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap().tobytes("png")

        all_text = ""
        for page in doc:
            all_text += (page.get_text() or "").upper()

        if "REITMANS" in all_text:
            brand = "REITMANS"

        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_page = (page.extract_text() or "").upper()

                if any(x in text_page for x in ["POLY CORE", "SEWING THREAD", "BUTTON"]):
                    continue

                tables = page.extract_tables()

                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)

                    if df.empty or len(df.columns) < 2:
                        continue

                    p_col, v_col = -1, -1

                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]

                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["DESCRIPTION", "POM NAME", "ITEM"]):
                                p_col = i

                        for i, cell in enumerate(row_up):
                            if i != p_col and any(k in cell for k in ["FINAL", "SPEC", "M"]):
                                v_col = i

                        if p_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_col]).strip().upper()

                                val = parse_val(d_row[v_col])

                                if val > 0:
                                    full_specs[name] = val
                            break

        return {"specs": full_specs, "img": img_bytes, "brand": brand}

    except:
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")

    try:
        res_c = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Đã nạp", res_c.count)
    except Exception as e:
        st.error(f"Lỗi DB: {e}")

    files = st.file_uploader("Upload Techpack", accept_multiple_files=True)

    if files and st.button("🚀 NẠP"):
        for f in files:
            d = extract_pom_deep_v462(f)

            if not d or not d['specs']:
                st.warning(f"Lỗi đọc: {f.name}")
                continue

            img_url, vec = "", []

            try:
                # ===== FIX STORAGE =====
                if d['img']:
                    path = f"lib/{f.name}.png"

                    supabase.storage.from_(BUCKET).upload(
                        path,
                        d['img'],   # ✅ FIX: truyền bytes trực tiếp
                        {"upsert": "true"}
                    )

                    pub = supabase.storage.from_(BUCKET).get_public_url(path)
                    img_url = pub.get("publicUrl", "")

                # ===== AI VECTOR =====
                if model_ai and d['img']:
                    img = Image.open(io.BytesIO(d['img'])).convert("RGB")

                    tf = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])

                    with torch.no_grad():
                        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()

            except Exception as e:
                st.warning(f"Lỗi ảnh: {e}")

            # ===== INSERT DB =====
            try:
                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec if vec else [],
                    "spec_json": d['specs'],
                    "image_url": img_url,
                    "category": d['brand']
                }).execute()

                st.success(f"✅ {f.name}")

            except Exception as e:
                st.error(f"Lỗi insert: {e}")

        st.rerun()

# ================= MAIN =================
st.title("🔍 AI Auditor")

t_file = st.file_uploader("Upload file kiểm", type="pdf")

if t_file:
    target = extract_pom_deep_v462(t_file)

    if target and target['specs']:
        st.success(f"Tìm thấy {len(target['specs'])} POM")

        db = supabase.table("ai_data").select("*").execute()

        if db.data:
            st.write("So sánh OK")
