import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from rapidfuzz import process, fuzz

# ================= 1. CẤU HÌNH HỆ THỐNG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96 PRO", page_icon="👖")

# CSS làm đẹp bảng giống như hình ảnh của bạn
st.markdown("""
    <style>
    .stTable { font-size: 14px; }
    thead th { background-color: #f0f2f6 !important; color: #333 !important; text-align: left !important; }
    td { text-align: left !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & XỬ LÝ SỐ ĐO =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    """Đọc số đo cực kỳ chính xác (số thập phân, phân số)"""
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        if not txt or any(x in txt for x in ["date", "page", "total", "size"]): return 0
        
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split(); return float(p[0]) + eval(p[1])
        return eval(v_str) if '/' in v_str else float(v_str)
    except: return 0

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().tolist()

# ================= 3. QUÉT PDF ĐA SIZE (CHỐNG LỆCH CỘT) =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, customer = {}, None, "UNKNOWN"
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        full_text = " ".join([p.get_text() for p in doc])
        cust_match = re.search(r"(?i)(CUSTOMER|BUYER|CLIENT)[:\s]+([^\n]+)", full_text)
        if cust_match: customer = cust_match.group(2).strip().upper()
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    n_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        if n_col == -1:
                            for i, v in enumerate(row):
                                if any(x in v for x in ["DESCRIPTION", "POM", "POSITION", "MEASUREMENT"]): n_col = i; break
                        for i, v in enumerate(row):
                            if i == n_col or not v or any(x in v for x in ["TOL", "+/-", "NO.", "STT"]): continue
                            if v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "2XL", "3XL"]): size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    if n_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 2 and val > 0: all_specs[s_name][pom] = val
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except: return None

# ================= 4. GIAO DIỆN & SIDEBAR =================
with st.sidebar:
    st.header("🏢 QUẢN LÝ KHO MẪU")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True)
    if new_files and st.button("XÁC NHẬN NẠP"):
        for f in new_files:
            d = extract_pdf_multi_size(f)
            if d:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_vector(d['img']),
                    "spec_json": d['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": d['customer']
                }).execute()
        st.success("Đã nạp kho thành công!"); st.rerun()

# ================= 5. ĐỐI SOÁT CHI TIẾT (SMART MATCHING) =================
st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload file PDF Audit", type="pdf")

if file_audit:
    with st.spinner("AI đang soi thông số..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            best = df_db.sort_values('sim', ascending=False).iloc[0]
            
            # Hiển thị ảnh so sánh
            c1, c2 = st.columns(2)
            with c1: st.image(target['img'], caption="Bản Audit hiện tại", use_container_width=True)
            with c2: st.image(best['image_url'], caption=f"Mẫu Gốc: {best['file_name']}", use_container_width=True)

            st.divider()
            st.subheader(f"📊 Đối soát chi tiết: {best['file_name']}")
            
            audit_all = target['all_specs']
            db_all = best['spec_json']
            
            sel_size = st.selectbox("Chọn Size:", list(audit_all.keys()))
            
            if sel_size in audit_all:
                spec_audit = audit_all[sel_size]
                # So khớp size mờ để lấy dữ liệu từ DB
                db_size_key = process.extractOne(sel_size, db_all.keys(), scorer=fuzz.Ratio)
                spec_ref = db_all.get(db_size_key[0], {}) if db_size_key else {}
                
                report = []
                for pom_audit, v_audit in spec_audit.items():
                    # So khớp POM mờ để tránh lệch ký tự (G010 vs G 010)
                    res_match = process.extractOne(pom_audit, spec_ref.keys(), scorer=fuzz.TokenSortRatio)
                    v_ref = spec_ref.get(res_match[0], 0) if res_match and res_match[1] > 80 else 0
                    
                    diff = round(v_audit - v_ref, 4)
                    status = "✅ OK" if abs(diff) < 0.0001 else f"❌ LỆCH ({diff:+.4f})"
                    
                    report.append({
                        "POM": pom_audit,
                        "Audit": v_audit,
                        "Gốc": v_ref,
                        "Lệch": diff,
                        "Kết quả": status
                    })
                
                df_res = pd.DataFrame(report)
                st.table(df_res) # Hiển thị bảng dạng tĩnh như hình mẫu
                
                # --- XUẤT EXCEL ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='Audit_Report')
                st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Audit_{best['file_name']}.xlsx")
    else:
        st.error("Không tìm thấy bảng thông số trong PDF này.")
