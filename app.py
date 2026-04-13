import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, json
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH (THAY THÔNG TIN CỦA BẠN) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

# Khởi tạo Supabase an toàn
@st.cache_resource
def get_supabase():
    return create_client(URL, KEY)

supabase = get_supabase()

st.set_page_config(layout="wide", page_title="AI Smart Auditor V112", page_icon="📏")

# --- KHỞI TẠO BIẾN SESSION AN TOÀN TUYỆT ĐỐI ---
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
    return [float(x) for x in vec] # Đảm bảo là list số thực

# ================= 3. TRÍCH XUẤT THÔNG MINH =================
def extract_pdf_v112(file):
    specs, img_bytes, category, customer = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                # Nén ảnh xuống chất lượng thấp hơn (matrix 1.0) để không bị treo khi upload
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.0, 1.0)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                t_up = (full_text + file.name).upper()
                
                # Nhận diện khách hàng
                if "REITMANS" in t_up: customer = "Reitmans"
                elif "VINEYARD VINES" in t_up or "WHALE" in t_up: customer = "Vineyard Vines"
                
                # Nhận diện loại hàng
                category = "QUẦN/VÁY" if any(x in t_up for x in ["PANT", "JEAN", "SKIRT", "QUẦN", "VÁY"]) else "ÁO"
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    txt_tb = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "POM NAME", "SPEC"] if k in txt_tb) < 2: continue
                    
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
                                # Parse số đơn giản
                                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(df.iloc[d_idx, v_col]))
                                val = float(nums[0]) if nums else 0
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# ================= 4. SIDEBAR - NẠP KHO (CHẾ ĐỘ DEBUG) =================
with st.sidebar:
    st.header("📂 KHO MASTER AI")
    try:
        res_c = supabase.table("ai_data").select("id", count="exact").execute()
        st.success(f"📊 Trong kho: **{res_c.count if res_c.count is not None else 0}** mẫu")
    except Exception as e:
        st.error(f"Lỗi hiển thị kho: {e}")

    st.divider()
    # Widget upload với Key an toàn
    new_files = st.file_uploader("Nạp Techpack Master", type="pdf", 
                                 accept_multiple_files=True, 
                                 key=f"up_widget_{st.session_state['up_id']}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
            success_count = 0
            for f in new_files:
                st.write(f"🔍 Đang phân tích: {f.name}")
                data = extract_pdf_v112(f)
                if data and data['specs']:
                    try:
                        # 1. Upload ảnh (Nén Matrix 1.0 giúp chạy nhanh)
                        path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        
                        # 2. Insert DB - Lấy đúng tên file làm prod_id
                        style_name = f.name.rsplit('.', 1)[0]
                        
                        st.write(f"💾 Đang lưu {style_name} vào Database...")
                        supabase.table("ai_data").insert({
                            "file_name": f.name, 
                            "customer": data['customer'],
                            "prod_id": style_name, 
                            "vector": get_vector(data['img']),
                            "spec_json": data['specs'],
                            "image_url": url,
                            "category": data['category']
                        }).execute()
                        success_count += 1
                    except Exception as e:
                        st.error(f"❌ Lỗi nạp {f.name}: {str(e)}") # Hiện lỗi đỏ chi tiết
                else:
                    st.warning(f"⚠️ Không tìm thấy bảng thông số trong {f.name}")
            
            if success_count > 0:
                status.update(label="✅ Đã nạp xong!", state="complete", expanded=False)
                st.session_state['up_id'] += 1
                st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V112")
# ... Phần Đối soát giữ nguyên logic của bản V111 ...
