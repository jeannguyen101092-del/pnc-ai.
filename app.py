import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, base64, difflib
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIGURATION & LOGO PPJ (BASE64 CHUẨN) =================
# Mã hóa Logo PPJ Group chuẩn để nhúng trực tiếp - Chắc chắn hiển thị 100%
PPJ_LOGO_B64 = "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"

# Cấu hình Supabase
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ Auditor Pro", page_icon="👔")

if 'master_key' not in st.session_state: st.session_state['master_key'] = 0
if 'audit_key' not in st.session_state: st.session_state['audit_key'] = 0
if 'selected_sku' not in st.session_state: st.session_state['selected_sku'] = None

# ================= 2. AI CORE & UTILS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes): return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. PDF LOGIC (STRICT) =================
def extract_pdf_multi_size(file_content):
    all_specs, img_bytes, is_reit = {}, None, False
    try:
        txt_all = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages[:2]: txt_all += (p.extract_text() or "").upper()
        if "REITMAN" in txt_all: is_reit = True

        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
        img_bytes = pix.tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    desc_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        for i, v in enumerate(row):
                            if is_reit and "POM NAME" in v: desc_col = i; break
                            elif not is_reit and ("DESCRIPTION" in v or "POM NAME" in v): desc_col = i; break
                        if desc_col != -1: break
                    if desc_col == -1: continue
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        for i, v in enumerate(row):
                            if i == desc_col or not v: continue
                            if any(x in v for x in ["TOL", "GRADE", "CODE", "+/-"]): continue
                            if len(v) <= 8 or v.isdigit() or v in ["XS","S","M","L","XL","NEW"]: size_cols[i] = v
                        if size_cols: break
                    if size_cols:
                        for s_col, s_name in size_cols.items():
                            temp_data = {}
                            for d_idx in range(len(df)):
                                pom_text = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                pom_text = re.sub(r'^\d+[\.\-\)]*\s*', '', pom_text)
                                if len(pom_text) < 3 or any(x in pom_text.upper() for x in ["DESCRIPTION", "POM NAME", "SIZE"]): continue
                                val = parse_val(df.iloc[d_idx, s_col])
                                if val > 0: temp_data[pom_text] = val
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        if not all_specs or img_bytes is None: return None
        return {"all_specs": all_specs, "img": img_bytes, "is_reit": is_reit}
    except: return None

# Thuật toán so khớp mờ nâng cao
def smart_match(query, choices):
    query_norm = re.sub(r'[^a-z0-9]', '', query.lower())
    best_m, max_s = None, 0
    for c in choices:
        c_norm = re.sub(r'[^a-z0-9]', '', c.lower())
        if query_norm == c_norm: return c
        score = difflib.SequenceMatcher(None, query_norm, c_norm).ratio()
        if score > max_s: max_s = score; best_m = c
    return best_m if max_s > 0.65 else None

# ================= 4. UI MASTER REPOSITORY =================
with st.sidebar:
    st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{PPJ_LOGO_B64}" width="200"></div>', unsafe_allow_html=True)
    st.markdown("---")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.title("📂 MASTER REPO")
    st.metric("Synchronized SKUs", f"{count}")
    
    st.divider()
    new_files = st.file_uploader("📥 Ingest Master Data", accept_
