import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor", page_icon="👖")

# Quản lý reset uploader để xóa file sau khi quét
if 'reset_key' not in st.session_state:
    st.session_state['reset_key'] = 0

# ================= 2. HÀM AI & HỖ TRỢ =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. TRÍCH XUẤT PDF (CHẶN DUNG SAI) =================
def extract_pdf_multi_size(file_content):
    all_specs, img_bytes, customer = {}, None, "UNKNOWN"
    try:
        txt_all = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages[:2]: txt_all += p.extract_text() or ""
        if "REIMANT" in txt_all.upper(): customer = "REIMANT"

        doc = fitz.open(stream=file_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue

                    desc_col, potential_size_cols = -1, {}
                    for r_idx in range(min(12, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        for i, v in enumerate(row):
                            if (customer == "REIMANT" and "POM NAME" in v) or ("DESCRIPTION" in v or "POM" in v):
                                desc_col = i
                        for i, v in enumerate(row):
                            if i == desc_col or not v: continue
                            if any(x in v for x in ["TOL", "+/-", "MIN", "MAX", "GRADE", "CODE"]): continue
                            if v.isdigit() or v in ["XS","S","M","L","XL","2XL","3XL"]:
                                potential_size_cols[i] = v

                    if desc_col != -1 and potential_size_cols:
                        for s_col, s_name in potential_size_cols.items():
                            temp_values, col_data = [], {}
                            for d_idx in range(len(df)):
                                desc = re.sub(r'^\d+[\.\-\)]*\s*', '', str(df.iloc[d_idx, desc_col])).strip()
                                if len(desc) < 3: continue
                                val = parse_val(df.iloc[d_idx, s_col])
                                if val > 0:
                                    col_data[desc] = val
                                    temp_values.append(val)
                            
                            # CHỈ LẤY CỘT NẾU TRUNG BÌNH > 1 (LOẠI DUNG SAI)
                            if temp_values and np.mean(temp_values) > 1.0:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(col_data)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.header("🏢 QUẢN LÝ KHO")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Số lượng mẫu trong kho", f"{res_count.count or 0} mẫu")
    
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("NẠP KHO"):
        for f in new_files:
            content = f.read()
            f_hash = get_file_hash(content)
            data = extract_pdf_multi_size(content)
            if data and data['all_specs']:
                path = f"lib_{f_hash}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").upsert({
                    "id": f_hash, "file_name": f.name, "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.session_state['reset_key'] += 1
        st.success("Đã nạp xong!"); st.rerun()

st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf", key=f"audit_{st.session_state['reset_key']}")

if file_audit:
    audit_content = file_audit.read()
    target = extract_pdf_multi_size(audit_content)
    
    # KIỂM TRA ĐỂ TRÁNH LỖI NAMEERROR
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            
            # HIỂN THỊ TOP 3
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            st.subheader("🎯 Mẫu tương đồng nhất")
            cols = st.columns(4)
            cols[0].image(target['img'], caption="FILE ĐANG QUÉT", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                cols[i+1].image(row['image_url'], caption=f"Top {i+1}: {row['sim']:.1%}", use_container_width=True)
                if cols[i+1].button(f"Chọn mẫu {i+1}", key=f"btn_{idx}"):
                    st.session_state['selected_ref'] = row.to_dict()

            best = st.session_state.get('selected_ref', top_3.iloc[0].to_dict())
            st.info(f"✅ Đối soát với: **{best['file_name']}**")
            
            sel_size = st.selectbox("Chọn Size:", list(target['all_specs'].keys()))
            spec_audit = target['all_specs'][sel_size]
            spec_ref = best['spec_json'].get(sel_size, list(best['spec_json'].values())[0])
            
            def norm(x): return re.sub(r'[^a-z0-9]', '', str(x).lower())
            ref_map = {norm(k): v for k, v in spec_ref.items()}
            
            report = []
            for pom, val in spec_audit.items():
                k_n = norm(pom); rv = ref_map.get(k_n, 0)
                if rv == 0:
                    for k, v in ref_map.items():
                        if k_n in k or k in k_n: rv = v; break
                diff = round(val - rv, 3)
                report.append({"Thông số": pom, "Thực tế": val, "Mẫu kho": rv, "Lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.2 else "❌ Lệch"})
            
            df_rep = pd.DataFrame(report)
            st.table(df_rep)
            
            # XUẤT EXCEL
            towrite = io.BytesIO()
            df_rep.to_excel(towrite, index=False, engine='xlsxwriter')
            st.download_button("📥 Xuất báo cáo Excel", data=towrite.getvalue(), file_name=f"Report_{file_audit.name}.xlsx")
            
            if st.button("Xóa và Quét lại"):
                st.session_state['reset_key'] += 1
                if 'selected_ref' in st.session_state: del st.session_state['selected_ref']
                st.rerun()
    else:
        st.error("⚠️ Không tìm thấy bảng số đo hợp lệ. Hãy kiểm tra lại file PDF.")
