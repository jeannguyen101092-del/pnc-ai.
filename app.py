import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH & KẾT NỐI =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor Gold", page_icon="👖")

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f8f9fa !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. HÀM AI & XỬ LÝ DỮ LIỆU =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    try:
        if not t: return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm)$', '', txt)
        if len(txt) > 10 or not txt or any(x in txt for x in ["date", "page", "total"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split(); val = float(p[0]) + eval(p[1])
        elif '/' in v_str: val = eval(v_str)
        else: val = float(v_str)
        return val if val <= 300 else 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. QUÉT PDF ĐA SIZE (CHỐNG NHẦM CỘT) =================
def extract_pdf_multi_size(file):
    all_specs = {}
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        
        # Tìm tên khách hàng
        full_text = " ".join([p.get_text() for p in doc])
        cust_match = re.search(r"(?i)(CUSTOMER|BUYER|CLIENT)[:\s]+([^\n]+)", full_text)
        customer = cust_match.group(2).strip().upper() if cust_match else "UNKNOWN"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 3: continue
                    
                    n_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        # Tìm cột POM
                        if n_col == -1:
                            for i, v in enumerate(row):
                                if any(x in v for x in ["DESCRIPTION", "POM", "MEASUREMENT"]): n_col = i; break
                        # Tìm cột Size (Loại bỏ cột No, Tolerance)
                        for i, v in enumerate(row):
                            if i == n_col or not v: continue
                            if any(x in v for x in ["TOL", "+/-", "NO.", "STT", "SIZE"]): continue
                            if v.isdigit() or any(s == v for s in ["XXS","XS","S","M","L","XL","XXL","3XL"]):
                                size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    
                    if n_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0: all_specs[s_name][pom] = val
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except Exception as e:
        st.error(f"Lỗi: {e}"); return None

# ================= 4. SIDEBAR QUẢN LÝ =================
with st.sidebar:
    st.header("🏢 KHO MẪU HỆ THỐNG")
    if st.button("Làm mới kho"): st.rerun()
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng mẫu", f"{res_count.count or 0}")
    except: pass
    
    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True)
    if new_files and st.button("XÁC NHẬN NẠP"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data and data['all_specs']:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": data['customer']
                }).execute()
        st.success("Đã nạp kho!"); st.rerun()

# ================= 5. LUỒNG ĐỐI SOÁT & XUẤT EXCEL =================
st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload file PDF Audit", type="pdf")

if file_audit:
    with st.spinner("AI đang phân tích..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            best = df_db.sort_values('sim', ascending=False).iloc[0]
            
            # Hiển thị ảnh so sánh
            c1, c2 = st.columns(2)
            c1.image(target['img'], caption="Bản Audit", use_container_width=True)
            c2.image(best['image_url'], caption=f"Mẫu Gốc (Khớp {best['sim']*100:.1f}%)", use_container_width=True)

            # Đối soát chi tiết
            st.divider()
            audit_specs = target['all_specs']
            db_specs = best['spec_json']
            
            sel_size = st.selectbox("Chọn Size đối soát:", list(audit_specs.keys()))
            
            if sel_size in db_specs:
                spec_audit, spec_ref = audit_specs[sel_size], db_specs[sel_size]
                rows = []
                for pom, v_audit in spec_audit.items():
                    v_ref = spec_ref.get(pom, 0)
                    diff = round(v_audit - v_ref, 3)
                    status = "✅ KHỚP" if abs(diff) < 0.126 else f"❌ LỆCH ({diff})"
                    rows.append({"POM": pom, "Audit": v_audit, "Gốc": v_ref, "Lệch": diff, "Kết quả": status})
                
                df_res = pd.DataFrame(rows)
                st.dataframe(df_res, use_container_width=True)
                
                # --- CHỨC NĂNG XUẤT EXCEL ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='Audit_Report')
                
                st.download_button(
                    label="📥 TẢI BÁO CÁO EXCEL",
                    data=output.getvalue(),
                    file_name=f"Audit_{best['file_name']}_{sel_size}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error(f"Size {sel_size} không có trong mẫu gốc!")
    else:
        st.warning("Không tìm thấy bảng thông số. Hãy kiểm tra lại định dạng PDF.")
