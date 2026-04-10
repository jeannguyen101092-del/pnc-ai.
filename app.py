import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V45.3", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except: return None
model_ai = load_ai()

# --- HÀM PARSE SỐ ĐO (CHUẨN REITMANS: PHÂN SỐ & THẬP PHÂN) ---
def parse_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ['nan', '-', 'none', 'tol']): return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0] # Lấy giá trị khớp đầu tiên
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT THÔNG SỐ ---
def extract_pom_v453(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").upper()
                if "POLY CORE" in text or "SEWING THREAD" in text: continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    desc_col, val_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        # Ưu tiên Reitmans (POM NAME)
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["POM NAME", "DESCRIPTION", "ITEM", "POINT OF MEASURE"]):
                                desc_col = i
                                break
                        # Ưu tiên Reitmans (NEW)
                        for i, cell in enumerate(row_up):
                            if i != desc_col and any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " M ", " L "]):
                                val_col = i
                                break
                        
                        if desc_col != -1 and val_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[desc_col]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["DATE", "PAGE", "NOTE"]): continue
                                val_num = parse_val(d_row[val_col])
                                if val_num > 0: full_specs[name] = val_num
        doc.close()
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR: NẠP FILE ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: st.error("Chưa kết nối Supabase")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            d = extract_pom_v453(f)
            if d and d['specs']:
                img_url = ""
                try:
                    path = f"lib_{f.name.replace('.pdf', '.png')}"
                    supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"x-upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                except: pass
                
                try:
                    supabase.table("ai_data").insert({"file_name": f.name, "spec_json": d['specs'], "image_url": img_url}).execute()
                    st.success(f"Nạp xong: {f.name} ({len(d['specs'])} dòng)")
                except Exception as e: st.error(f"Lỗi DB: {e}")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V45.3")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v453(t_file)
    if target and target['specs']:
        st.success(f"✅ Tìm thấy {len(target['specs'])} thông số từ file kiểm.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu với: {m['file_name']}")
            
            c1, c2 = st.columns(2)
            if target['img']:
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
            if m['image_url']:
                with c2: st.image(m['image_url'], caption="Mẫu gốc trong kho")

            diff_list = []
            for p_name, v_target in target['specs'].items():
                v_ref = 0
                p_clean = clean_pos(p_name)
                for k_ref, val_ref in m['spec_json'].items():
                    if p_clean == clean_pos(k_ref):
                        v_ref = val_ref
                        break
                diff_list.append({
                    "Hạng mục": p_name, 
                    "Kiểm tra (NEW)": v_target, 
                    "Mẫu gốc (REF)": v_ref, 
                    "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"
                })
            
            if diff_list:
                df_r = pd.DataFrame(diff_list)
                st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                
                # --- KHÔI PHỤC NÚT XUẤT EXCEL ---
                out = io.BytesIO()
                df_r.to_excel(out, index=False)
                st.download_button(
                    label="📥 Tải báo cáo Excel",
                    data=out.getvalue(),
                    file_name=f"Audit_{m['file_name']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if st.button("🗑️ Xóa kết quả để quét file mới"):
    st.session_state.au_key += 1
    st.rerun()
