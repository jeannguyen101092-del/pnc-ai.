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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V115", page_icon="📏")

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

# ================= 3. LOGIC QUÉT CẠN (DEEP SCAN) =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if '/' in txt: # Xử lý phân số 1/2, 1/4...
            p = re.findall(r'\d+', txt)
            if len(p) == 2: return float(p[0])/float(p[1])
            if len(p) == 3: return float(p[0]) + (float(p[1])/float(p[2]))
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def extract_pdf_v115(file):
    specs, img_bytes, category, customer = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        
        # 1. Lấy ảnh và Text tổng quát
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.0, 1.0)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                t_up = (full_text + file.name).upper()
                if "REITMANS" in t_up: customer = "Reitmans"
                elif "VINEYARD VINES" in t_up: customer = "Vineyard Vines"
                category = "QUẦN/VÁY" if any(x in t_up for x in ["PANT", "JEAN", "SKIRT", "QUẦN", "VÁY"]) else "ÁO"

        # 2. CHẾ ĐỘ QUÉT CẠN: Thử mọi bảng có trong PDF
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if len(df) < 5 or len(df.columns) < 2: continue # Bỏ qua bảng quá nhỏ

                    # Tìm cột tên và cột số liệu dựa trên nội dung (không dựa trên Header)
                    n_col, v_col = -1, -1
                    
                    # Thử quét 10 dòng đầu để tìm cột nào chứa chữ, cột nào chứa số
                    for r_idx in range(min(10, len(df))):
                        row = [str(x).upper() for x in df.iloc[r_idx]]
                        # Nếu cột có "POM" hoặc "NAME" hoặc "DESC"
                        for i, val in enumerate(row):
                            if any(k in val for k in ["POM", "NAME", "DESC", "POSITION"]): n_col = i
                            if any(k in val for k in ["NEW", "SPEC", "SAMP", "32", "M", "S"]): v_col = i
                    
                    # Fallback: Nếu không tìm thấy header, mặc định cột 0 là tên, cột cuối là số
                    if n_col == -1: n_col = 0
                    if v_col == -1: v_col = len(df.columns) - 1

                    # Tiến hành bốc dữ liệu
                    for i in range(len(df)):
                        name = str(df.iloc[i, n_col]).strip().upper()
                        if len(name) < 4 or any(x in name for x in ["TOL", "REF", "REMARK", "COMMENT"]): continue
                        
                        val = parse_val(df.iloc[i, v_col])
                        if val > 0: specs[name] = val
                    
                    if len(specs) > 3: break # Nếu đã lấy được kha khá thông số thì dừng
                if len(specs) > 3: break
                
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except Exception as e:
        return None

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
                                 key=f"up_v115_{st.session_state['up_id']}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        with st.status("Đang quét dữ liệu...", expanded=True) as status:
            success_count = 0
            for f in new_files:
                data = extract_pdf_v115(f)
                if data and len(data['specs']) > 0:
                    try:
                        path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        
                        style_code = f.name.split('.')[0] # Lấy tên file làm mã Style
                        
                        supabase.table("ai_data").insert({
                            "file_name": f.name, "customer": data['customer'],
                            "prod_id": style_code, "vector": get_vector(data['img']),
                            "spec_json": data['specs'], "image_url": url, "category": data['category']
                        }).execute()
                        success_count += 1
                    except Exception as e:
                        st.error(f"Lỗi lưu trữ {f.name}: {e}")
                else:
                    st.warning(f"⚠️ Vẫn không thấy bảng trong {f.name}. Thử kiểm tra lại file PDF.")
            
            if success_count > 0:
                status.update(label="✅ Nạp kho thành công!", state="complete")
                st.session_state['up_id'] += 1
                st.rerun()

# ================= 5. MAIN =================
st.title("🔍 AI SMART AUDITOR - V115")
# ... Phần Đối soát ...
