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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V95", page_icon="📏")

if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f0f2f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & CÔNG CỤ QUÉT KHÁCH HÀNG =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def extract_customer_name(text):
    patterns = [r"(?i)CUSTOMER[:\s]+([^\n]+)", r"(?i)CLIENT[:\s]+([^\n]+)", r"(?i)BUYER[:\s]+([^\n]+)", r"(?i)KHÁCH HÀNG[:\s]+([^\n]+)"]
    for p in patterns:
        match = re.search(p, text)
        if match: return match.group(1).strip().upper()
    return "UNKNOWN"

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {"VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM", "GOWN"], "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM", "QUẦN"], "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "COAT", "ÁO", "SWEATER"]}
    scores = {cat: sum(t.count(k) for k in keys) for cat, keys in keywords.items()}
    detected = max(scores, key=scores.get)
    return detected if scores[detected] > 0 else "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 10 or not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs", "date"]): return 0
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
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT THÔNG MINH (DÒ THEO SIZE) =================
def extract_pdf_v95(file, target_size=None):
    specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        category, customer = detect_category(full_text, file.name), extract_customer_name(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    n_col, v_col = -1, -1
                    
                    # QUAN TRỌNG: Dò hàng tiêu đề để tìm cột Size chính xác
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # Tìm cột POM
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): n_col = i
                        
                        # Tìm cột trùng với Size yêu cầu (ví dụ tìm cột "10")
                        if target_size:
                            for i, v in enumerate(row_up):
                                if str(target_size).strip().upper() == v: 
                                    v_col = i; break
                        
                        if n_col != -1 and (v_col != -1 or target_size is None): break
                    
                    # Nếu không tìm thấy cột size cụ thể, lấy cột mặc định
                    if n_col != -1 and v_col == -1:
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

# ================= 4. SIDEBAR (NẠP KHO) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    size_save = st.text_input("Size mặc định khi nạp kho (VD: 10)", "10")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            data = extract_pdf_v95(f, target_size=size_save)
            if data and data['specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                        "category": data['category'], "customer_name": data['customer']
                    }).execute()
                except Exception as e: st.error(f"Lỗi nạp {f.name}: {e}")
        st.success("Nạp thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. LUỒNG ĐỐI SOÁT CHÍNH (HÌNH ẢNH AI + PRIORITY) =================
st.title("🔍 AI SMART AUDITOR - V95")

# Giao diện chính để chọn file và size
c1, c2 = st.columns([3, 1])
with c1: file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")
with c2: size_audit = st.text_input("Nhập Size cần check (VD: 10)", "10")

if file_audit:
    with st.spinner("Đang trích xuất và tìm mẫu tương đồng..."):
        target = extract_pdf_v95(file_audit, target_size=size_audit)
    
    if target and target["specs"]:
        st.info(f"✨ Khách hàng: **{target['customer']}** | Đang lấy thông số **Size {size_audit}**")
        
        # 1. Lấy dữ liệu từ kho (cùng phân loại)
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # 2. TÍNH ĐỘ TƯƠNG ĐỒNG HÌNH ẢNH (Hàm AI gốc của bạn)
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
            
            # 3. LOGIC ƯU TIÊN KHÁCH HÀNG TP MỚI
            def get_prio(name):
                name = str(name).upper()
                if "TP MỚI" in name: return 2
                if name == target['customer']: return 1
                return 0
            df_db['priority'] = df_db['customer_name'].apply(get_prio)
            
            # Sắp xếp: Ưu tiên TP MỚI -> Rồi mới đến độ giống hình ảnh AI
            df_db = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False])
            
            best_match = df_db.iloc[0]
            
            # 4. HIỂN THỊ KẾT QUẢ
            st.success(f"Mẫu khớp nhất: **{best_match['file_name']}** (Khách: {best_match['customer_name']})")
            
            col_img1, col_img2 = st.columns(2)
            with col_img1: st.image(target['img'], caption="Bản đang Audit", use_container_width=True)
            with col_img2: st.image(best_match['image_url'], caption="Mẫu gốc đối chiếu (AI Match)", use_container_width=True)

            # Bảng so sánh thông số
            lib_specs = best_match['spec_json']
            audit_table = []
            for pom, val_new in target['specs'].items():
                val_old = lib_specs.get(pom, 0)
                diff = val_new - val_old
                status = "✅ Khớp" if abs(diff) < 0.25 else f"❌ Lệch ({diff:+.2f})"
                audit_table.append({"Vị trí đo (POM)": pom, "Mẫu Gốc (Lib)": val_old, f"Audit (Size {size_audit})": val_new, "Kết quả": status})
            
            st.table(pd.DataFrame(audit_table))
        else:
            st.warning("Không tìm thấy mẫu nào cùng phân loại trong kho.")
