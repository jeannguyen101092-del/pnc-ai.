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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V93", page_icon="📏")

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & IMAGE PROCESSING =================
@st.cache_resource
def load_model():
    # Sử dụng ResNet18 để lấy đặc trưng (Feature Vector)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

def get_image_vector(img_bytes):
    """Chuyển ảnh từ PDF thành vector 512 chiều"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vector = model_ai(preprocess(img).unsqueeze(0)).flatten().numpy()
    return vector

# ================= 3. TRÍCH XUẤT DỮ LIỆU PDF =================
def parse_val(t):
    """Xử lý các con số đo đạc, kể cả phân số như 1/2, 3/4"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_pdf_v93(file, customer="Auto"):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Lấy ảnh trang 1 làm mẫu nhận diện
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text += page.get_text()
        doc.close()
        
        category = "KHÁC"
        t = (full_text + " " + file.name).upper()
        if any(x in t for x in ["PANT", "JEAN", "QUẦN"]): category = "QUẦN"
        elif any(x in t for x in ["SHIRT", "TOP", "ÁO"]): category = "ÁO"
        elif any(x in t for x in ["DRESS", "VÁY"]): category = "VÁY/ĐẦM"

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    # Kiểm tra xem có phải bảng thông số không (Dựa vào từ khóa đo đạc)
                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if not any(x in flat_text for x in ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER"]): continue
                    
                    # Tìm cột chứa tên thông số (Description) và cột chứa giá trị (Spec/New/Sample)
                    n_col, v_col = -1, -1
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row if c]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i
                            if any(x in v for x in ["NEW", "SPEC", "SAMP", "M", "S", "L"]): v_col = i
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).strip().upper()
                                val = parse_val(df.iloc[d_idx, v_col])
                                if len(name) > 3 and val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

# ================= 4. GIAO DIỆN CHÍNH =================
st.title("🔍 AI SMART FASHION AUDITOR")

# Sidebar: Quản lý kho dữ liệu
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    uploaded_files = st.file_uploader("Nạp Techpack vào kho", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Nạp vào hệ thống"):
        for f in uploaded_files:
            data = extract_pdf_v93(f)
            if data:
                vec = get_image_vector(data['img']).tolist()
                path = f"lib_{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                    "image_url": img_url, "category": data['category']
                }).execute()
        st.success("Đã cập nhật kho!")

# Main Flow: Đối soát
file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    target = extract_pdf_v93(file_audit)
    if target:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(target['img'], caption="Ảnh mẫu đang kiểm tra", use_column_width=True)
        
        with col2:
            st.subheader(f"Thông tin trích xuất: {target['category']}")
            # Tìm kiếm trong kho mẫu cùng loại
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            
            if res.data:
                target_vec = get_image_vector(target['img']).reshape(1, -1)
                matches = []
                for item in res.data:
                    sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))[0][0]
                    matches.append({**item, "similarity": sim})
                
                # Lấy mẫu có độ tương đồng cao nhất
                best_match = max(matches, key=lambda x: x['similarity'])
                
                st.info(f"Mẫu khớp nhất trong kho: **{best_match['file_name']}** (Độ giống: {best_match['similarity']:.2%})")
                
                # SO SÁNH THÔNG SỐ
                st.write("### 📊 BẢNG ĐỐI SOÁT THÔNG SỐ")
                audit_rows = []
                target_specs = target['specs']
                master_specs = best_match['spec_json']
                
                for key, val in target_specs.items():
                    # Tìm key tương ứng trong Master (dùng fuzzy match đơn giản hoặc exact match)
                    master_val = master_specs.get(key, 0)
                    diff = val - master_val if master_val else 0
                    status = "✅ KHỚP" if abs(diff) < 0.2 else f"❌ LỆCH {diff:+}"
                    audit_rows.append({
                        "Vị trí đo (POM)": key,
                        "File Hiện Tại": val,
                        "File Gốc (Master)": master_val if master_val else "N/A",
                        "Kết quả": status
                    })
                
                df_audit = pd.DataFrame(audit_rows)
                st.table(df_audit)
            else:
                st.warning("Không tìm thấy mẫu cùng loại trong kho để đối soát.")

