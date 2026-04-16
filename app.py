import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os

# ================= 1. CONFIGURATION =================
# Link dự phòng nếu bạn chưa up file logo.png lên GitHub
BACKUP_LOGO = "https://githubusercontent.com" 

URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

# Cấu hình trang phải nằm ở dòng đầu tiên
st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state:
    st.session_state['reset_key'] = 0

# Hàm hiển thị Logo thông minh
def display_logo(width=200):
    if os.path.exists("logo.png"):
        st.image("logo.png", width=width)
    else:
        # Nếu chưa có file logo.png trên GitHub, hiển thị chữ PPJ tạm thời
        st.markdown(f"<h1 style='color: #1E3A8A;'>PPJ GROUP</h1>", unsafe_allow_html=True)

# ================= 2. AI CORE ENGINE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
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

# ================= 3. PDF EXTRACTION =================
def extract_pdf_multi_size(file_content):
    all_specs, img_bytes, is_reit = {}, None, False
    try:
        txt_check = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages[:1]: txt_check += (p.extract_text() or "").upper()
        if "REITMAN" in txt_check: is_reit = True

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
                            if len(v) <= 8 or v.isdigit() or v in ["XS","S","M","L","XL"]: size_cols[i] = v
                        if size_cols: break
                    if size_cols:
                        for s_col, s_name in size_cols.items():
                            temp_data = {}
                            for d_idx in range(len(df)):
                                pom_text = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                if len(pom_text) < 3 or any(x in pom_text.upper() for x in ["DESCRIPTION", "POM NAME", "SIZE"]): continue
                                val = parse_val(df.iloc[d_idx, s_col])
                                if val > 0: temp_data[pom_text] = val
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes, "is_reit": is_reit}
    except: return None

# ================= 4. UI PPJ GROUP =================
with st.sidebar:
    display_logo(width=220)
    st.markdown("---")
    st.title("📂 MASTER REPOSITORY")
    
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Synchronized SKUs", f"{count} Models")
    
    used_mb = (count * 0.15)
    percent = min((used_mb / 1024) * 100, 100.0)
    st.write(f"💾 **Cloud Storage:** {used_mb:.1f}MB / 1GB")
    st.progress(percent / 100)

    st.divider()
    st.subheader("📥 Data Ingestion")
    new_files = st.file_uploader("Upload Tech-Packs (Bulk)", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE DATABASE", use_container_width=True):
        with st.spinner("AI Processing..."):
            for f in new_files:
                c = f.read(); h = get_file_hash(c)
                
                # CHỐNG TRÙNG: Kiểm tra ID tồn tại chưa
                exist = supabase.table("ai_data").select("id").eq("id", h).execute()
                if len(exist.data) > 0: continue
                
                data = extract_pdf_multi_size(c)
                # CHỐNG FILE LỖI: Phải có cả hình (img) và thông số (all_specs)
                if data and data.get('img') and data.get('all_specs'):
                    path = f"lib_{h}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
        st.session_state['reset_key'] += 1
        st.rerun()

# Header chính
h_col1, h_col2 = st.columns([1, 4])
with h_col1:
    display_logo(width=120)
with h_col2:
    st.title("AI SMART AUDITOR PRO")
    st.markdown("*Premium Technical Audit System for PPJ Group*")

st.markdown("---")

file_audit = st.file_uploader("📤 Drag & Drop Tech-Pack for Auditing", type="pdf", key=f"audit_{st.session_state['reset_key']}")

if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_multi_size(a_bytes)
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            
            st.subheader(f"🎯 AI Best Matches")
            cols = st.columns(4)
            with cols[0]:
                st.image(target['img'], caption="SOURCE FILE", use_container_width=True)
            
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i+1]:
                    st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}", use_container_width=True)
                    if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['sel'] = row.to_dict()

            best = st.session_state.get('sel', top_3.iloc[0].to_dict())
            st.success(f"**REFERENCE SKU:** {best['file_name']}")
            # --- PHẦN BỔ SUNG: NÚT XUẤT EXCEL ---
            st.divider()
            st.subheader("📊 Export Results")
            
            # Chuyển đổi dữ liệu thông số (spec_json) của mẫu đang chọn thành DataFrame
            if 'spec_json' in best:
                # Tạo danh sách dữ liệu để xuất
                export_data = []
                for size, poms in best['spec_json'].items():
                    for pom_name, value in poms.items():
                        export_data.append({"Size": size, "Description/POM": pom_name, "Value": value})
                
                df_export = pd.DataFrame(export_data)
                
                # Tạo buffer để lưu file Excel trong bộ nhớ
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_export.to_excel(writer, index=False, sheet_name='TechPack_Specs')
                
                # Hiển thị nút bấm tải về
                st.download_button(
                    label="📥 DOWNLOAD SPEC TO EXCEL",
                    data=buffer.getvalue(),
                    file_name=f"Spec_{best['file_name']}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
