import streamlit as st
import io, pdfplumber, re, pandas as pd
from supabase import create_client, Client

# --- CONFIG (BẮT BUỘC: Thay đúng URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45.4")

# --- HÀM ĐỌC SỐ REITMANS & CHUNG ---
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

# --- TRÍCH XUẤT THÔNG SỐ SIÊU TỐC ---
def extract_pom_fast(pdf_file):
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
                                val = parse_val(d_row[v_idx])
                                if val > 0: full_specs[name] = val
                            break
        return full_specs
    except: return None

# --- SIDEBAR: NẠP DỮ LIỆU ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        # Lấy số lượng thực tế từ DB
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: st.error("Chưa kết nối Supabase")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            specs = extract_pom_fast(f)
            if specs:
                try:
                    # NẠP TỐI GIẢN: Chỉ nạp những gì chắc chắn bảng ai_data đang có
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "spec_json": specs
                    }).execute()
                    st.success(f"Đã nạp: {f.name}")
                except Exception as e:
                    st.error(f"Lỗi DB: {e}")
            else:
                st.warning(f"Không quét được bảng trong {f.name}")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V45.4")
t_file = st.file_uploader("Upload file đối soát", type="pdf")

if t_file:
    target_specs = extract_pom_fast(t_file)
    if target_specs:
        st.success(f"✅ Tìm thấy {len(target_specs)} thông số.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu với: {m['file_name']}")
            diff_list = [{"Hạng mục": k, "Kiểm tra": v, "Gốc": m['spec_json'].get(k, 0), "Lệch": round(v - m['spec_json'].get(k, 0), 2) if m['spec_json'].get(k, 0) > 0 else "N/A"} for k, v in target_specs.items()]
            st.table(pd.DataFrame(diff_list))
