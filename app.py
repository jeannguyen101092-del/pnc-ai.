import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
from supabase import create_client, Client

# --- CONFIG (BẮT BUỘC: Thay URL và KEY của bạn vào đây) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45.3", page_icon="📊")

# --- HÀM PARSE SỐ (GIỮ CHUẨN REITMANS) ---
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
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
def extract_pom_deep_v453(pdf_file):
    full_specs = {}
    try:
        pdf_content = pdf_file.read()
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_idx, v_idx = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["POM NAME", "DESCRIPTION", "ITEM"]): p_idx = i
                            if any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " M "]): v_idx = i
                        
                        if p_idx != -1 and v_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 2: continue
                                val = parse_reitmans_val(d_row[v_idx])
                                if val > 0: full_specs[name] = val
                            break
        return full_specs
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# --- SIDEBAR (THANH % VÀ BẮT LỖI DB) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except Exception as e:
        st.error("Chưa kết nối được Supabase. Kiểm tra lại URL/KEY.")
        st.exception(e)

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            specs = extract_pom_deep_v453(f)
            if specs:
                try:
                    # NẠP THỬ: Chỉ nạp 2 cột cơ bản nhất để kiểm tra kết nối
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "spec_json": specs
                    }).execute()
                    st.toast(f"Nạp xong: {f.name}")
                except Exception as db_e:
                    # HIỆN LỖI ĐỎ NẾU THẤT BẠI
                    st.error(f"❌ LỖI DATABASE khi nạp {f.name}")
                    st.exception(db_e)
            else:
                st.warning(f"Không tìm thấy bảng POM trong file {f.name}")
            p_bar.progress((i + 1) / len(files))
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V45.3")
t_file = st.file_uploader("Upload file đối soát", type="pdf")

if t_file:
    target_specs = extract_pom_deep_v453(t_file)
    if target_specs:
        st.success(f"✅ Tìm thấy {len(target_specs)} thông số.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu với: {m['file_name']}")
            diff_list = []
            for name, val in target_specs.items():
                ref_val = m['spec_json'].get(name, 0)
                diff_list.append({
                    "Hạng mục": name, "Kiểm tra": val, 
                    "Mẫu gốc": ref_val, "Lệch": round(val - ref_val, 2) if ref_val > 0 else "N/A"
                })
            st.table(pd.DataFrame(diff_list))
