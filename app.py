import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, json
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

@st.cache_resource
def get_supabase():
    return create_client(URL, KEY)
supabase = get_supabase()

st.set_page_config(layout="wide", page_title="AI Smart Auditor V116", page_icon="📏")

# Khởi tạo biến Session an toàn
if 'up_id' not in st.session_state:
    st.session_state['up_id'] = 0

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return [float(x) for x in vec]

# ================= 3. TRÍCH XUẤT THÔNG MINH =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums) if nums else 0
    except: return 0

def extract_pdf_v116(file):
    specs, img_bytes, category, customer = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.0, 1.0)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                t_up = (full_text + file.name).upper()
                if "REITMANS" in t_up: customer = "Reitmans"
                elif "VINEYARD VINES" in t_up: customer = "Vineyard Vines"
                category = "QUẦN/VÁY" if any(x in t_up for x in ["PANT", "JEAN", "SKIRT", "QUẦN", "VÁY"]) else "ÁO"
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    # Quét tìm bảng POM
                    if len(df) < 5: continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if customer == "Reitmans" and "POM NAME" in row_up:
                            n_col = row_up.index("POM NAME")
                            v_col = next((i for i, v in enumerate(row_up) if any(x in v for x in ["NEW", "SAMPLE", "SPEC"])), -1)
                        else:
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i; break
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["NEW", "SPEC", "M", "32"]): v_col = i; break
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or "TOL" in name: continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None

# ================= 4. SIDEBAR - NẠP KHO =================
with st.sidebar:
    st.header("📂 KHO MASTER AI")
    try:
        res_c = supabase.table("ai_data").select("id", count="exact").execute()
        st.success(f"📊 Trong kho: **{res_c.count if res_c.count is not None else 0}** mẫu")
    except: st.info("Đang kết nối kho...")

    st.divider()
    new_files = st.file_uploader("Nạp Techpack Master", type="pdf", 
                                 accept_multiple_files=True, 
                                 key=f"up_v116_{st.session_state['up_id']}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        with st.status("Đang nạp dữ liệu...", expanded=True) as status:
            success_count = 0
            for f in new_files:
                data = extract_pdf_v116(f)
                if data and data['specs']:
                    try:
                        # Upload ảnh
                        path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        
                        # Lấy mã Style từ tên file
                        style_code = f.name.split('.')
                        
                        # INSERT (Đảm bảo tên cột khớp 100% với DB)
                        supabase.table("ai_data").insert({
                            "file_name": f.name, 
                            "customer": data['customer'],
                            "prod_id": style_code, 
                            "vector": get_vector(data['img']),
                            "spec_json": data['specs'], 
                            "image_url": url, 
                            "category": data['category']
                        }).execute()
                        success_count += 1
                    except Exception as e:
                        st.error(f"Lỗi DB {f.name}: {e}")
                else:
                    st.warning(f"Không tìm thấy bảng thông số trong {f.name}")
            
            if success_count > 0:
                status.update(label="✅ Nạp kho thành công!", state="complete")
                st.session_state['up_id'] += 1
                st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V116")
# ... Phần Đối soát tiếp nối bên dưới ...
