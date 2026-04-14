import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
# Thay đổi URL và KEY của bạn tại đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V95", page_icon="📏")

if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f0f2f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & CÔNG CỤ HỖ TRỢ =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# --- TÍNH NĂNG THÊM: QUÉT TÊN KHÁCH HÀNG ---
def extract_customer_name(text):
    patterns = [
        r"(?i)CUSTOMER[:\s]+([^\n]+)", 
        r"(?i)CLIENT[:\s]+([^\n]+)", 
        r"(?i)BUYER[:\s]+([^\n]+)",
        r"(?i)KHÁCH HÀNG[:\s]+([^\n]+)"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match: return match.group(1).strip().upper()
    return "UNKNOWN"

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {
        "VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM", "GOWN"],
        "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM", "QUẦN"],
        "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "COAT", "ÁO", "SWEATER"]
    }
    scores = {cat: sum(t.count(k) for k in keys) for cat, keys in keywords.items()}
    detected = max(scores, key=scores.get)
    return detected if scores[detected] > 0 else "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 10 or not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs", "date", "2024", "2025"]): 
            return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        val = float(parts[0]) + eval(parts[1]) if ' ' in v_str and (parts := v_str.split()) else (eval(v_str) if '/' in v_str else float(v_str))
        return val if val <= 200 else 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
    return vec.astype(float).tolist()

# ================= 3. TRÍCH XUẤT THÔNG MINH (DÒ CỘT SỐ) =================
def extract_pdf_v95(file):
    specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        category = detect_category(full_text, file.name)
        # Quét tên khách hàng
        customer = extract_customer_name(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER", "THIGH", "RISE"] if k in flat_text) < 2: continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): n_col = i; break
                        if n_col != -1: break
                    if n_col != -1:
                        max_nums = 0
                        for i in range(len(df.columns)):
                            if i == n_col: continue
                            num_count = sum(1 for val in df.iloc[:12, i] if parse_val(val) > 0)
                            if num_count > max_nums: max_nums = num_count; v_col = i
                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if len(name) > 3 and val > 0 and not any(x in name for x in ["DESCRIPTION", "POM"]):
                                specs[name] = val
                if specs: break 
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None

# ================= 4. SIDEBAR (QUẢN LÝ KHO) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{res_db.count or 0} file")
    except: st.error("Lỗi kết nối database.")

    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới vào kho", accept_multiple_files=True, key=f"uploader_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            with st.spinner(f"Đang nạp {f.name}..."):
                data = extract_pdf_v95(f)
                if data and data['specs']:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                        "image_url": url, "category": data['category'],
                        "customer_name": data['customer'] # Lưu thêm khách hàng
                    }).execute()
        st.success("Nạp thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. LUỒNG ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR - V95")

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    with st.spinner("Đang trích xuất và đối chiếu..."):
        target = extract_pdf_v95(file_audit)
    
    if target and target["specs"]:
        st.info(f"✨ Khách hàng: **{target['customer']}** | Phân loại: **{target['category']}**")
        
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # GIỮ NGUYÊN: Tính độ tương đồng hình ảnh (AI Vector)
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
            
            # THÊM TÍNH NĂNG: Logic ưu tiên TP MỚI
            def get_priority(name):
                name = str(name).upper()
                if "TP MỚI" in name: return 2
                if name == target['customer']: return 1
                return 0
            
            df_db['priority'] = df_db['customer_name'].apply(get_priority)
            
            # Sắp xếp: Ưu tiên trước (Priority), tương đồng hình ảnh sau (sim_score)
            df_db = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False])
            
            best_match = df_db.iloc[0]
            
            # Hiển thị kết quả so sánh
            st.success(f"Mẫu khớp nhất: **{best_match['file_name']}** (Khách: {best_match['customer_name']})")
            
            col1, col2 = st.columns(2)
            with col1: st.image(target['img'], caption="Bản Audit", use_container_width=True)
            with col2: st.image(best_match['image_url'], caption="Mẫu Gốc", use_container_width=True)

            # Bảng thông số chi tiết
            st.subheader("📊 Bảng đối soát thông số")
            lib_specs = best_match['spec_json']
            audit_table = []
            for pom, val_audit in target['specs'].items():
                val_lib = lib_specs.get(pom, 0)
                diff = val_audit - val_lib
                status = "✅ Khớp" if abs(diff) < 0.25 else f"❌ Lệch ({diff:+.2f})"
                audit_table.append({"POM": pom, "Mẫu Gốc": val_lib, "Audit": val_audit, "Kết quả": status})
            st.table(pd.DataFrame(audit_table))
