import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs" 
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V83", page_icon="🛡️")

# ================= 2. MODEL AI & PHÂN LOẠI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text):
    t = str(text).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "HOODIE", "SWEATER"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN"]): return "VÁY/ĐẦM"
    return "KHÁC"

def ultra_clean(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper().strip())

# ================= 3. HÀM TRÍCH XUẤT NÂNG CẤP V83 =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def is_measurement_table(df):
    all_text = " ".join(df.astype(str).values.flatten()).upper()
    # Mở rộng từ khóa nhận diện bảng thông số
    measurement_keywords = ["WAIST", "HIP", "RISE", "THIGH", "KNEE", "LEG", "INSEAM", "LENGTH", "CHEST", "SHOULDER", "NECK", "SLEEVE"]
    score = sum(1 for word in measurement_keywords if word in all_text)
    return score >= 1 # Chỉ cần thấy ít nhất 1 từ khóa ngành may là chấp nhận

def extract_pdf_v83(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            # Lấy ảnh trang 1 làm thumbnail
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
            for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    if not is_measurement_table(df): continue

                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(20).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        # Tìm cột Description
                        if any(x in " ".join(row_up) for x in ["DESCRIPTION", "DESC", "POM", "MEASURE"]):
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESCRIPTION", "DESC", "NAME", "POM"]): n_col = i; break
                            # Tìm cột giá trị
                            for i, v in enumerate(row_up):
                                if any(target == v or target in v for target in ["NEW", "SAMPLE", "SPEC", "M", "32", "34", "S", "L"]):
                                    if i != n_col: v_col = i; break
                            
                            if n_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                    if len(name) < 3 or any(x in name for x in ["TOL", "REF", "TOTAL"]): continue
                                    val = parse_val(d_row[v_col])
                                    if 0 < val < 500: # Nới lỏng giới hạn lên 500 (phù hợp cả đơn vị cm)
                                        specs[name] = val
                                break
                if specs: break # Đã tìm thấy bảng ở trang này thì không quét các trang sau của cùng 1 file
        
        if not specs or img_bytes is None: return None
        return {"specs": specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi phân tích: {e}")
        return None

# ================= 4. SIDEBAR (QUẢN LÝ KHO) =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    try:
        res_db = supabase.table("ai_data").select("file_name", "category").execute()
        data_lib = res_db.data if res_db.data else []
        st.info(f"Kho hiện tại: {len(data_lib)} file")
    except:
        st.error("Chưa kết nối Supabase!")
        data_lib = []

    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới", type="pdf", accept_multiple_files=True)
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        for f in new_files:
            # 1. Kiểm tra trùng tên file
            if any(d['file_name'] == f.name for d in data_lib):
                st.warning(f"⏩ Đã có: {f.name}"); continue
            
            with st.spinner(f"Đang phân tích {f.name}..."):
                data = extract_pdf_v83(f)
                
                # 2. Báo lỗi cụ thể nếu không nạp được
                if not data:
                    st.error(f"❌ Loại bỏ {f.name}: Không tìm thấy bảng thông số may mặc (Cần có cột Desc/Size).")
                    continue
                
                # 3. Tạo Vector AI và nạp
                try:
                    img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                    
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                        "image_url": url, "category": data['category']
                    }).execute()
                    st.toast(f"✅ Đã nạp thành công: {f.name}")
                except Exception as db_e:
                    st.error(f"Lỗi Database với file {f.name}: {db_e}")
        st.rerun()

# ================= 5. MAIN (GIỮ NGUYÊN) =================
# ... (Phần Main giống V82) ...
