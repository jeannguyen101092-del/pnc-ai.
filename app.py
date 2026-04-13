import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG (Thay bằng thông tin của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V93", page_icon="📏")

# CSS làm đẹp
st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f0f2f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & UTILS =================
@st.cache_resource
def load_model():
    # Sử dụng ResNet18 để lấy đặc trưng ảnh
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

def detect_category(text, filename=""):
    """Nhận diện loại hàng thông minh"""
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "HOODIE", "ÁO"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "VÁY", "ĐẦM"]): return "VÁY/ĐẦM"
    return "KHÁC"

def parse_val(t):
    """Xử lý số đo bao gồm cả phân số (1 1/2)"""
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

def get_image_vector(img_bytes):
    """Chuyển ảnh thành Vector để so sánh độ giống nhau"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vector = model_ai(tf(img).unsqueeze(0)).flatten().cpu().detach().numpy()
    return vector.astype(float).tolist()

# ================= 3. HÀM TRÍCH XUẤT PDF =================
def extract_pdf_v93(file):
    specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: 
                full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        category = detect_category(full_text, file.name)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    # Kiểm tra bảng có phải bảng thông số (POM) không
                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER"] if k in flat_text) < 2: 
                        continue

                    n_col, v_col = -1, -1
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # Tìm cột tên thông số
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): n_col = i; break
                        # Tìm cột giá trị (Size M, New, Spec...)
                        for i, v in enumerate(row_up):
                            if i != n_col and any(x in v for x in ["NEW", "SAMPLE", "SPEC", "M", "S", "L", "32", "34"]): 
                                v_col = i; break
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, v_col])
                                if len(name) > 3 and val > 0: specs[name] = val
                            break
                if specs: break 
        return {"specs": specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        # Lấy số lượng file thực tế
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        count_val = res_db.count if res_db.count is not None else 0
        st.metric("Số mẫu hiện có", f"{count_val} file")
        data_lib = res_db.data if res_db.data else []
    except: 
        st.error("Lỗi kết nối Supabase")
        data_lib = []

    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới vào kho", type="pdf", accept_multiple_files=True)
    
    if new_files and st.button("🚀 Nạp vào hệ thống"):
        for f in new_files:
            with st.spinner(f"Đang xử lý {f.name}..."):
                data = extract_pdf_v93(f)
                if data and data['specs']:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{str(f.name).replace(' ', '_')}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    supabase.table("ai_data").insert({
                        "file_name": str(f.name), "vector": vec, 
                        "spec_json": data['specs'], "image_url": url, 
                        "category": data['category']
                    }).execute()
        st.success("Đã cập nhật kho!")
        st.rerun()

# ================= 5. MAIN FLOW =================
st.title("🔍 AI SMART FASHION AUDITOR")

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    with st.spinner("Đang đối soát dữ liệu..."):
        target = extract_pdf_v93(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Phát hiện: **{target['category']}** | {len(target['specs'])} vị trí đo.")
        
        # Lấy dữ liệu cùng chủng loại để so sánh
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if res.data:
            # So sánh Vector tìm mẫu giống nhất
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            matches = []
            for item in res.data:
                sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))[0][0]
                matches.append({**item, "similarity": sim})
            
            best_match = max(matches, key=lambda x: x['similarity'])
            
            col_img1, col_img2 = st.columns(2)
            with col_img1: st.image(target['img'], caption="File mới Upload")
            with col_img2: st.image(best_match['image_url'], caption=f"Mẫu khớp nhất ({best_match['similarity']:.1%})")

            # Bảng so sánh thông số
            st.subheader(f"📊 Bảng đối soát với: {best_match['file_name']}")
            audit_data = []
            for pom, val in target["specs"].items():
                master_val = best_match['spec_json'].get(pom, 0)
                diff = round(val - master_val, 3) if master_val else "N/A"
                
                status = "✅ Khớp"
                if master_val == 0: status = "❓ Thiếu thông tin"
                elif abs(diff) > 0.125: status = f"❌ Lệch ({diff})"
                
                audit_data.append({
                    "Vị trí đo (POM)": pom,
                    "Thông số mới": val,
                    "Thông số gốc": master_val,
                    "Kết quả": status
                })
            
            st.table(pd.DataFrame(audit_data))
        else:
            st.warning(f"Trong kho chưa có mẫu nào thuộc loại **{target['category']}**.")
    else:
        st.error("Không tìm thấy bảng thông số kỹ thuật trong file này.")

st.divider()
st.caption("AI Fashion Auditor V93 - Hỗ trợ kiểm tra sai lệch thông số tự động.")
