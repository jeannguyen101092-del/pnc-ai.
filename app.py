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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V86", page_icon="🛡️")

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text):
    t = str(text).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "ÁO"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "VÁY", "ĐẦM"]): return "VÁY/ĐẦM"
    return "KHÁC"

def ultra_clean(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper().strip())

# ================= 3. HÀM TRÍCH XUẤT "THOÁNG" HƠN V86 =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_pdf_v86(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text)
        
        all_candidate_tables = []

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Đếm số lượng ô chứa giá trị số
                    num_count = 0
                    for col in df.columns[1:]:
                        num_count += df[col].apply(lambda x: 1 if parse_val(x) > 0 else 0).sum()
                    
                    if num_count > 2: # Chỉ cần có trên 2 số là ghi nhận bảng tiềm năng
                        all_candidate_tables.append((num_count, df))

        if not all_candidate_tables:
            return "ERR_NO_TABLE"

        # Lấy bảng có nhiều số đo nhất
        all_candidate_tables.sort(key=lambda x: x[0], reverse=True)
        df = all_candidate_tables[0][1]

        n_col, v_col = -1, -1
        # Tìm cột Tên và cột Giá trị linh hoạt
        for r_idx, row in df.head(20).iterrows():
            row_up = [str(c).upper().strip() for c in row if c]
            for i, v in enumerate(row_up):
                if any(x in v for x in ["DESC", "POM", "NAME", "VỊ TRÍ", "TÊN"]): n_col = i; break
            
            for i, v in enumerate(row_up):
                if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "M", "32", "34", "S", "L"]):
                    if i != n_col: v_col = i; break
            
            if n_col != -1 and v_col != -1:
                for d_idx in range(r_idx + 1, len(df)):
                    d_row = df.iloc[d_idx]
                    name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                    if len(name) < 3 or any(x in name for x in ["TOL", "REF"]): continue
                    val = parse_val(d_row[v_col])
                    if val > 0: specs[name] = val
                break
        
        if not specs: return "ERR_NO_SPECS"
        return {"specs": specs, "img": img_bytes, "category": category}
    except Exception as e:
        return f"ERR_SYSTEM: {str(e)}"

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    try:
        res_db = supabase.table("ai_data").select("file_name").execute()
        st.info(f"Kho hiện tại: {len(res_db.data) if res_db.data else 0} file")
    except: st.error("Chưa kết nối Supabase!")

    st.divider()
    st.subheader("🚀 NẠP TECHPACK MỚI")
    new_files = st.file_uploader("Kéo thả PDF vào đây", type="pdf", accept_multiple_files=True)
    
    if st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        if not new_files:
            st.warning("Vui lòng chọn file trước!")
        else:
            for f in new_files:
                with st.spinner(f"Đang phân tích {f.name}..."):
                    result = extract_pdf_v86(f)
                    
                    if isinstance(result, str):
                        if result == "ERR_NO_TABLE": st.error(f"❌ {f.name}: Không thấy bảng nào.")
                        elif result == "ERR_NO_SPECS": st.error(f"❌ {f.name}: Thấy bảng nhưng không thấy số đo.")
                        else: st.error(f"❌ {f.name}: {result}")
                        continue
                    
                    # Nạp kho
                    try:
                        img = Image.open(io.BytesIO(result['img'])).convert('RGB')
                        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                        vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                        
                        path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                        supabase.storage.from_(BUCKET).upload(path, result['img'], {"upsert":"true", "content-type": "image/png"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": result['specs'], "image_url": url, "category": result['category']}).execute()
                        st.toast(f"✅ Đã nạp thành công: {f.name}")
                    except Exception as e:
                        st.error(f"❌ Lỗi Database {f.name}: {e}")
            st.rerun()

# ================= 5. MAIN (GIỮ NGUYÊN SO SÁNH) =================
st.title("🔍 AI SMART AUDITOR - V86")
# ... (Phần Main so sánh của V85 giữ nguyên) ...
