import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Hãy kiểm tra kỹ URL và KEY) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V45.7", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

# --- HÀM PARSE SỐ REITMANS (GIỮ NGUYÊN LOGIC CŨ) ---
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
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT THÔNG SỐ (VÉT SẠCH TRANG POM) ---
def extract_pom_v457(pdf_file):
    full_specs = {}
    try:
        pdf_content = pdf_file.read()
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").upper()
                if "POLY CORE" in text: continue # Né trang BOM
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    p_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        for i, c in enumerate(row_up):
                            if any(k in c for k in ["POM NAME", "DESCRIPTION", "ITEM"]): p_col = i
                        for i, c in enumerate(row_up):
                            if i != p_col and any(k in c for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " M "]): v_col = i
                        
                        if p_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_col]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["DATE", "PAGE"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: full_specs[name] = val
                            break
        return {"specs": full_specs}
    except: return None

# --- SIDEBAR: NẠP FILE (CÓ THANH TIẾN ĐỘ %) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: st.error("Lỗi kết nối database")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        progress_bar = st.progress(0) # Khởi tạo thanh %
        for idx, f in enumerate(files):
            d = extract_pom_v457(f)
            if d and d['specs']:
                try:
                    # Nạp thẳng dữ liệu Text vào DB
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "spec_json": d['specs']
                    }).execute()
                    st.toast(f"Đã nạp: {f.name}")
                except Exception as e:
                    st.error(f"Lỗi nạp file {f.name}: {e}")
            
            # Cập nhật % chạy
            progress_bar.progress((idx + 1) / len(files))
        
        st.success("Hoàn thành quá trình nạp!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V45.7")
t_file = st.file_uploader("Upload file đối soát (PDF)", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v457(t_file)
    if target and target['specs']:
        st.success(f"✅ Tìm thấy {len(target['specs'])} thông số từ file kiểm.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            m = db_res.data[-1] # Lấy mẫu mới nhất nạp vào
            st.subheader(f"✨ Đối chiếu với: {m['file_name']}")
            
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
                
                # XUẤT EXCEL
                out = io.BytesIO()
                df_r.to_excel(out, index=False)
                st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Audit_{m['file_name']}.xlsx")
