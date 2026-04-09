import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay thông tin của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.2", page_icon="📊")

# ================= INIT SUPABASE =================
@st.cache_resource
def init_supabase():
    try:
        return create_client(URL, KEY)
    except:
        return None

supabase = init_supabase()

# ================= AI MODEL =================
@st.cache_resource
def load_model():
    # Sử dụng ResNet50 để lấy Feature Vector (bỏ lớp phân loại cuối)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= HELPER FUNCTIONS =================
def get_vector(img_bytes):
    """Chuyển đổi ảnh sang Vector 2048 chiều"""
    try:
        if not img_bytes: return None
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

def extract_techpack(pdf_file):
    """Trích xuất ảnh mẫu và thông số từ PDF"""
    specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        # 1. Lấy ảnh đại diện trang 1 bằng PyMuPDF (fitz)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        # 2. Lấy thông số bảng biểu bằng pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables: continue
                for tb in tables:
                    for row in tb:
                        if len(row) >= 2 and row[0] and row[-1]:
                            key = str(row[0]).strip().upper()
                            val = re.findall(r"\d+\.?\d*", str(row[-1]))
                            if val: specs[key] = val[0]
        
        return {"spec": specs, "img": img_bytes}
    except Exception as e:
        st.error(f"Lỗi đọc file {pdf_file.name}: {e}")
        return None

def load_samples():
    """Tải dữ liệu từ Supabase Database"""
    try:
        res = supabase.table("ai_data").select("*").execute()
        return res.data if res.data else []
    except:
        return []

# ================= INTERFACE =================
samples = load_samples()

# SIDEBAR: QUẢN LÝ DỮ LIỆU
with st.sidebar:
    st.header("📦 Kho dữ liệu")
    st.metric("Số mẫu hiện có", len(samples))
    
    files = st.file_uploader("Upload Techpacks (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if files and st.button("🚀 Nạp vào hệ thống"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            data = extract_techpack(f)
            if data and data['img']:
                name = f.name.replace(".pdf","")
                img_path = f"{name}_{i}.png"
                
                # FIX LỖI TYPEERROR: Chỉ định rõ tham số path và file
                try:
                    supabase.storage.from_(BUCKET).upload(
                        path=img_path, 
                        file=data['img'], 
                        file_options={"content-type": "image/png"}, 
                        upsert=True
                    )
                    img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
                    
                    vec = get_vector(data['img'])
                    
                    supabase.table("ai_data").upsert({
                        "file_name": name,
                        "vector": vec,
                        "spec_json": data['spec'],
                        "image_url": img_url
                    }).execute()
                except Exception as upload_err:
                    st.error(f"Lỗi tại file {f.name}: {upload_err}")
            
            p_bar.progress((i + 1) / len(files))
        st.success("Hoàn tất nạp dữ liệu!")
        st.rerun()

# MAIN AREA: KIỂM TRA MẪU (AUDIT)
st.title("🔍 AI FASHION AUDITOR V34.2")
test_file = st.file_uploader("Kéo tệp PDF cần kiểm tra vào đây", type=["pdf"], key="main_audit")

if test_file:
    with st.spinner("Đang phân tích AI..."):
        test_data = extract_techpack(test_file)
        
        if test_data and test_data['img']:
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.image(test_data['img'], caption="Ảnh trích xuất từ PDF kiểm tra", use_container_width=True)
                test_vec = get_vector(test_data['img'])
            
            # So sánh Similarity
            if samples and test_vec:
                results = []
                for s in samples:
                    # Tính toán độ tương đồng Cosine
                    sim = cosine_similarity([test_vec], [s['vector']])[0][0]
                    results.append({"data": s, "sim": sim})
                
                # Lấy mẫu khớp nhất
                best = max(results, key=lambda x: x['sim'])
                
                with col2:
                    st.subheader("Kết quả đối soát AI")
                    sim_pct = round(best['sim'] * 100, 1)
                    st.write(f"Mẫu khớp nhất: **{best['data']['file_name']}**")
                    st.write(f"Độ tương đồng AI: **{sim_pct}%**")
                    st.progress(float(best['sim']))
                    
                    # So sánh bảng thông số
                    st.divider()
                    st.write("📊 **So sánh thông số kỹ thuật (Specs):**")
                    
                    s_audit = test_data['spec']
                    s_db = best['data']['spec_json']
                    
                    comparison = []
                    for k, v in s_audit.items():
                        if k in s_db:
                            diff = round(float(v) - float(s_db[k]), 2)
                            status = "✅ OK" if abs(diff) < 0.2 else "❌ LỆCH"
                            comparison.append({
                                "Hạng mục": k,
                                "Mẫu kiểm": v,
                                "Mẫu gốc": s_db[k],
                                "Chênh lệch": diff,
                                "Kết quả": status
                            })
                    
                    if comparison:
                        st.table(pd.DataFrame(comparison))
                    else:
                        st.info("Không tìm thấy các hạng mục tương ứng để so sánh tự động.")
        else:
            st.error("Không thể đọc được ảnh từ file PDF này. Vui lòng kiểm tra lại định dạng file.")

# FOOTER
st.markdown("---")
st.caption("Fashion Auditor AI System - Version 34.2 (Stable)")
