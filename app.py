# AI FASHION AUDITOR V34.2 - FIXED VERSION
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
# Thay đổi URL và KEY thật của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.2", page_icon="📊")

# ================= INIT =================
@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

try:
    supabase = init_supabase()
except:
    st.error("Chưa cấu hình Supabase URL/KEY!")

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= VECTOR =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
            return vec.tolist()
    except Exception as e:
        st.error(f"Lỗi tạo vector: {e}")
        return None

# ================= EXTRACT PDF =================
def extract_techpack(pdf_file):
    specs, img_bytes, raw = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Lấy ảnh trang đầu làm mẫu
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for tb in tables:
                    if len(tb) < 2: continue
                    for row in tb:
                        if not row or not row[0]: continue
                        desc = str(row[0]).strip().upper()
                        # Logic tìm số đơn giản
                        nums = re.findall(r"\d+\.?\d*", str(row[-1]))
                        if nums: specs[desc] = nums[0]
        
        return {"spec": specs, "img": img_bytes}
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# ================= MAIN APP =================
def load_samples():
    try:
        res = supabase.table("ai_data").select("*").execute()
        return res.data if res.data else []
    except:
        return []

samples = load_samples()

# SIDEBAR: UPLOAD & DATABASE
with st.sidebar:
    st.header("📦 Kho dữ liệu")
    st.metric("Số mẫu trong kho", len(samples))
    
    uploaded_files = st.file_uploader("Upload Techpacks (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Nạp vào hệ thống"):
        progress_bar = st.progress(0)
        for idx, f in enumerate(uploaded_files):
            d = extract_techpack(f)
            if d:
                name = f.name.replace(".pdf","")
                # Upload ảnh
                img_path = f"{name}.png"
                supabase.storage.from_(BUCKET).upload(img_path, d['img'], 
                    file_options={"content-type":"image/png"}, upsert=True)
                img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
                
                # Tạo vector & Lưu DB
                vec = get_vector(d['img'])
                supabase.table("ai_data").upsert({
                    "file_name": name,
                    "vector": vec,
                    "spec_json": d['spec'],
                    "image_url": img_url
                }).execute()
            progress_bar.progress((idx + 1) / len(uploaded_files))
        st.success("Đã nạp xong!")
        st.rerun()

# MAIN INTERFACE: AUDIT
st.title("🔍 AI FASHION AUDITOR V34.2")

test_file = st.file_uploader("Kéo tệp PDF cần kiểm tra vào đây", type=["pdf"], key="checker")

if test_file:
    col1, col2 = st.columns([1, 2])
    test_data = extract_techpack(test_file)
    
    if test_data:
        with col1:
            st.image(test_data['img'], caption="Ảnh từ file kiểm tra", use_container_width=True)
            test_vec = get_vector(test_data['img'])
        
        # Tìm mẫu tương đồng nhất
        if samples and test_vec:
            best_match = None
            max_sim = 0
            
            for s in samples:
                sim = cosine_similarity([test_vec], [s['vector']])[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = s
            
            with col2:
                st.subheader(f"Kết quả phân tích AI")
                sim_percent = round(max_sim * 100, 2)
                st.write(f"**Độ tương đồng:** {sim_percent}%")
                st.progress(max_sim)
                
                if best_match:
                    st.write(f"✅ Mẫu khớp nhất: **{best_match['file_name']}**")
                    
                    # So sánh thông số (Specs)
                    st.write("### So sánh thông số kỹ thuật")
                    df_comp = []
                    s1 = test_data['spec']
                    s2 = best_match['spec_json']
                    
                    for k in s1:
                        if k in s2:
                            diff = float(s1[k]) - float(s2[k])
                            status = "✅ Khớp" if abs(diff) < 0.5 else "❌ Lệch"
                            df_comp.append({"Hạng mục": k, "Mẫu kiểm": s1[k], "Kho dữ liệu": s2[k], "Chênh lệch": diff, "Trạng thái": status})
                    
                    if df_comp:
                        st.table(pd.DataFrame(df_comp))
                    else:
                        st.warning("Không tìm thấy thông số chung để so sánh.")
    else:
        st.error("Không thể trích xuất dữ liệu từ file này.")
