import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os

# ================= 1. CONFIGURATION =================
BACKUP_LOGO = "https://githubusercontent.com" 

URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state:
    st.session_state['reset_key'] = 0

def display_logo(width=200):
    if os.path.exists("logo.png"):
        st.image("logo.png", width=width)
    else:
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
        # ... (giữ nguyên phần check text và is_reit của bạn) ...
        txt_check = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages[:1]: txt_check += (p.extract_text() or "").upper()
        if "REITMAN" in txt_check: is_reit = True

        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        
        # --- LOGIC MỚI: CHỈ LẤY HÌNH VẼ (SKETCH) ---
        # Tìm tất cả các đối tượng là hình ảnh hoặc đường kẻ vẽ trong file PDF
        paths = page.get_drawings() 
        if paths:
            # Xác định vùng bao quanh tất cả các nét vẽ (thường là Sketch)
            bbox = page.rect # Mặc định lấy cả trang
            # Thử tìm vùng tập trung các nét vẽ để crop
            x0, y0, x1, y1 = page.rect
            # Thông thường hình vẽ Techpack nằm ở giữa hoặc bên trái trang 1
            # Chúng ta sẽ crop bỏ bớt phần Header và Footer (nơi chứa nhiều chữ nhất)
            crop_rect = fitz.Rect(x0, y0 + 100, x1, y1 - 150) 
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), clip=crop_rect)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
            
        img_bytes = pix.tobytes("png")
        doc.close()
        # ... (giữ nguyên phần trích xuất table phía dưới của bạn) ...


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
                # CHỐNG TRÙNG
                exist = supabase.table("ai_data").select("id").eq("id", h).execute()
                if len(exist.data) > 0: continue
                
                data = extract_pdf_multi_size(c)
                # CHỐNG FILE LỖI
                if data and data.get('img') and data.get('all_specs'):
                    path = f"lib_{h}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
        st.session_state['reset_key'] += 1
        st.rerun()

h_col1, h_col2 = st.columns([1, 4])
with h_col1:
    display_logo(width=120)
with h_col2:
    st.title("AI SMART AUDITOR PRO")
    st.markdown("*Premium Technical Audit System for PPJ Group*")

st.markdown("---")

file_audit = st.file_uploader("📤 Drag & Drop Tech-Pack for Auditing", type="pdf", key=f"audit_{st.session_state['reset_key']}")

# ================= PHẦN SỬA LOGIC SO SÁNH (Dán đè vào đoạn xử lý so sánh) =================
if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_multi_size(a_bytes)
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            
            # 1. Tính toán độ tương đồng hình ảnh (Cũ)
            db_vectors = np.array([v for v in df_db['vector']])
            df_db['sim'] = cosine_similarity(t_vec, db_vectors).flatten()

            # 2. LOGIC MỚI: KIỂM TRA TỪ KHÓA ĐỂ PHÂN LOẠI (Chống quần short so quần dài)
            # Lấy tên file đang audit để tìm từ khóa chính
            t_name = file_audit.name.upper()
            def calculate_text_score(row_name):
                row_name = str(row_name).upper()
                score = 0
                # Phân loại độ dài quần
                if ("SHORT" in t_name) == ("SHORT" in row_name): score += 0.2
                if ("LONG" in t_name) == ("LONG" in row_name): score += 0.2
                # Phân loại loại túi/chi tiết
                keywords = ["POCKET", "PATCH", "WELT", "JOGGER", "CARGO"]
                for kw in keywords:
                    if (kw in t_name) == (kw in row_name): score += 0.1
                return score

            df_db['text_bonus'] = df_db['file_name'].apply(calculate_text_score)
            
            # Kết hợp điểm hình ảnh và điểm từ khóa (Ưu tiên hình ảnh nhưng có lọc từ khóa)
            df_db['final_sim'] = df_db['sim'] + df_db['text_bonus']
            
            # Sắp xếp theo điểm tổng hợp mới
            top_3 = df_db.sort_values('final_sim', ascending=False).head(3)
            
            # (Phần hiển thị UI bên dưới giữ nguyên như cũ của bạn...)
            st.subheader(f"🎯 AI Best Matches")
            cols = st.columns(4)
            # ... tiếp tục code hiển thị cũ ...

            with cols[0]:
                st.image(target['img'], caption="SOURCE FILE", use_container_width=True)
            
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i+1]:
                    st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}", use_container_width=True)
                    if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['sel'] = row.to_dict()

            best = st.session_state.get('sel', top_3.iloc[0].to_dict())
            st.success(f"**REFERENCE SKU:** {best['file_name']}")

            # --- HIỂN THỊ BẢNG THÔNG SỐ (PHẦN BẠN CẦN) ---
            st.subheader("📋 Measurement Comparison")
            for size_name, specs in target['all_specs'].items():
                with st.expander(f"SIZE: {size_name}", expanded=True):
                    ref_specs = best['spec_json'].get(size_name, {})
                    rows = []
                    for pom, val in specs.items():
                        ref_val = ref_specs.get(pom, 0)
                        diff = val - ref_val
                        color = "red" if abs(diff) > 0.25 else "green"
                        rows.append({
                            "Measurement Point": pom,
                            "Target": val,
                            "Reference": ref_val,
                            "Diff": f"{diff:+.3f}"
                        })
                    st.table(pd.DataFrame(rows))

            # --- NÚT XUẤT EXCEL ---
            st.divider()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(rows).to_excel(writer, index=False, sheet_name='Audit_Report')
            st.download_button(label="📥 EXPORT AUDIT TO EXCEL", data=output.getvalue(), file_name=f"Audit_{best['file_name']}.xlsx", mime="application/vnd.ms-excel")
