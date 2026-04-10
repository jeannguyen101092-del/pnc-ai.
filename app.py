import streamlit as st
import io, pdfplumber, re, pandas as pd, numpy as np
from supabase import create_client, Client

# --- CONFIG (BẮT BUỘC: Điền đúng URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V46.0", page_icon="📊")

# --- HÀM PARSE SỐ (GIỮ CHUẨN PHÂN SỐ REITMANS) ---
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

# --- TRÍCH XUẤT THÔNG SỐ (QUÉT CẠN TRANG POM) ---
def extract_pom_v460(pdf_file):
    full_specs = {}
    try:
        pdf_content = pdf_file.read()
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_pg = (page.extract_text() or "").upper()
                # Né trang phụ liệu (BOM)
                if any(x in text_pg for x in ["POLY CORE", "SEWING THREAD", "BUTTON"]): continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        # Tìm cột Tên: Thêm Description cho Express/Walmart
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["POM NAME", "DESCRIPTION", "ITEM"]):
                                p_col = i
                                break
                        # Tìm cột Số đo: Thêm Final/Size/12 cho đa dòng hàng
                        for i, cell in enumerate(row_up):
                            if i != p_col and any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " 12 ", " M "]):
                                v_col = i
                                break
                        
                        if p_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_col]).replace('\n',' ').strip().upper()
                                if len(name) < 2 or any(x in name for x in ["DATE", "PAGE", "NOTE"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: full_specs[name] = val
                            break
        return full_specs
    except: return None

# --- SIDEBAR: NẠP FILE (FIX TREO) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        # Kiểm tra kết nối và đếm số lượng file trong DB
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except Exception as conn_err:
        st.error(f"Lỗi kết nối: {conn_err}")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for idx, f in enumerate(files):
            specs = extract_pom_v460(f)
            if specs:
                try:
                    # NẠP TỐI GIẢN (Chỉ nạp các cột chắc chắn tồn tại để không bị crash)
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "spec_json": specs
                    }).execute()
                    st.toast(f"Đã nạp: {f.name}")
                except Exception as db_err:
                    st.error(f"Lỗi nạp DB ({f.name}): {db_err}")
            else:
                st.warning(f"Không quét được bảng trong {f.name}")
            p_bar.progress((idx + 1) / len(files))
        
        # Buộc app load lại để cập nhật số lượng
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V46.0")
t_file = st.file_uploader("Upload file đối soát (PDF)", type="pdf")

if t_file:
    target_specs = extract_pom_v460(t_file)
    if target_specs:
        st.success(f"✅ Quét được {len(target_specs)} thông số từ file đối soát.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            # Lấy mẫu mới nhất nạp vào kho để đối chiếu
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu với: {m['file_name']}")
            
            diff_list = []
            for p_name, v_target in target_specs.items():
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
                
                # XUẤT EXCEL BÁO CÁO
                out = io.BytesIO()
                df_r.to_excel(out, index=False)
                st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Report_{m['file_name']}.xlsx")
