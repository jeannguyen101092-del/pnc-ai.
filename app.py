import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V95 - Priority", page_icon="📏")

if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

# Giao diện CSS
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    .priority-tag { background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 12px; }
    thead th { background-color: #f0f2f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & TRÍCH XUẤT TÊN KHÁCH HÀNG =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def extract_customer_name(text):
    """Tìm kiếm tên khách hàng dựa trên từ khóa phổ biến trong ngành may"""
    patterns = [
        r"(?i)CUSTOMER[:\s]+([^\n]+)", 
        r"(?i)CLIENT[:\s]+([^\n]+)", 
        r"(?i)BUYER[:\s]+([^\n]+)",
        r"(?i)KHÁCH HÀNG[:\s]+([^\n]+)"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            # Lấy tên và làm sạch (bỏ các ký tự dư thừa)
            name = match.group(1).strip().upper()
            return name[:50] # Giới hạn độ dài
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
        if ' ' in v_str:
            parts = v_str.split()
            val = float(parts[0]) + eval(parts[1])
        else:
            val = eval(v_str) if '/' in v_str else float(v_str)
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

# ================= 3. TRÍCH XUẤT PDF TỔNG HỢP =================
def extract_pdf_v95(file):
    specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Chụp ảnh trang 1 làm vector nhận diện
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        category = detect_category(full_text, file.name)
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

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{res_db.count or 0} file")
    except: st.error("Lỗi kết nối database.")

    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            with st.spinner(f"Đang xử lý {f.name}..."):
                data = extract_pdf_v95(f)
                if data and data['specs']:
                    try:
                        vec = get_image_vector(data['img'])
                        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))
                        path = f"lib_{clean_name}.png"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        
                        supabase.table("ai_data").insert({
                            "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                            "image_url": url, "category": data['category'],
                            "customer_name": data['customer'] # Đã thêm lưu khách hàng
                        }).execute()
                    except Exception as e:
                        st.error(f"Lỗi khi nạp {f.name}: {e}")
        
        st.success("Đã cập nhật kho dữ liệu!")
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. LUỒNG ĐỐI SOÁT & ƯU TIÊN =================
st.title("🔍 AI SMART AUDITOR V95")

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    with st.spinner("Đang phân tích file..."):
        target = extract_pdf_v95(file_audit)
    
    if target and target["specs"]:
        st.info(f"✨ Khách hàng: **{target['customer']}** | Phân loại: **{target['category']}** | {len(target['specs'])} thông số.")
        
        # Lấy toàn bộ dữ liệu mẫu để so sánh
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # Tính toán độ tương đồng hình ảnh
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs)[0]
            
            # --- LOGIC ƯU TIÊN ---
            # 1. Ưu tiên khách hàng chứa chữ "TP MỚI"
            # 2. Hoặc khách hàng trùng chính xác với tên trong file đang audit
            df_db['priority'] = df_db['customer_name'].apply(
                lambda x: 2 if "TP MỚI" in str(x).upper() 
                else (1 if str(x).upper() == target['customer'] else 0)
            )
            
            # Sắp xếp: Ưu tiên trước, sau đó tới độ giống hình ảnh
            df_db = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False])
            best_match = df_db.iloc[0]
            
            # Giao diện hiển thị
            c1, c2 = st.columns(2)
            with c1:
                st.image(target['img'], caption="Bản đang kiểm tra", use_container_width=True)
            with c2:
                label_prio = "🔥 ƯU TIÊN: TP MỚI" if best_match['priority'] == 2 else "✅ Mẫu tương đồng"
                st.image(best_match['image_url'], caption=f"{label_prio} ({best_match['file_name']})", use_container_width=True)
                st.write(f"Khách hàng trong kho: **{best_match['customer_name']}**")

            # Bảng so sánh
            st.subheader("📊 Kết quả đối soát chi tiết")
            lib_specs = best_match['spec_json']
            comparison = []
            for pom, val_audit in target['specs'].items():
                val_lib = lib_specs.get(pom, 0)
                diff = abs(val_audit - val_lib)
                status = "✅ KHỚP" if diff < 0.25 else f"❌ LỆCH ({diff:.2f})"
                comparison.append({"Thông số (POM)": pom, "Thư viện mẫu": val_lib, "Bản Audit": val_audit, "Kết quả": status})
            
            st.table(pd.DataFrame(comparison))
        else:
            st.warning("Kho dữ liệu đang trống. Vui lòng nạp mẫu vào Sidebar.")

# Cuối file
