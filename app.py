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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96", page_icon="🏢")

if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    .percentage { color: #007bff; font-weight: bold; font-size: 22px; }
    thead th { background-color: #f8f9fa !important; color: #333 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & CÔNG CỤ HỖ TRỢ =================
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

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 10 or not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs", "date"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        val = float(v_str.split()[0]) + eval(v_str.split()[1]) if ' ' in v_str else (eval(v_str) if '/' in v_str else float(v_str))
        return val if val <= 250 else 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TỰ ĐỘNG QUÉT ĐA SIZE (ALL SIZES) =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        customer = extract_customer_name(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty: continue
                    
                    # 1. Tìm dòng tiêu đề để xác định cột POM và danh sách Size
                    n_col = -1
                    size_cols = {} # {col_index: size_name}
                    
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): 
                                n_col = i
                            # Nếu cột có tiêu đề là số hoặc chữ viết tắt Size (S, M, L, XL...)
                            elif len(v) > 0 and (v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "XXL", "2XL", "3XL"])):
                                size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    
                    # 2. Trích xuất dữ liệu cho từng size
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0 and pom != "DESCRIPTION":
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

# ================= 4. GIAO DIỆN VÀ SIDEBAR =================
# Thống kê kho mẫu ngay phần đầu Sidebar
with st.sidebar:
    st.header("🏢 QUẢN LÝ KHO MẪU")
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số file trong kho", f"{count_res.count or 0} mẫu")
    except: st.write("Đang kết nối database...")
    
    st.divider()
    st.subheader("🚀 Nạp mẫu mới (Tự quét đa size)")
    new_files = st.file_uploader("Upload Techpack gốc", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data and data['all_specs']:
                vec = get_image_vector(data['img'])
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, 
                    "spec_json": data['all_specs'], # Lưu toàn bộ JSON đa size
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": data['customer']
                }).execute()
        st.success("Đã nạp kho thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. LUỒNG ĐỐI SOÁT TỰ ĐỘNG =================
st.title("🔍 AI SMART AUDITOR V96 - AUTO SIZE MATCHING")

file_audit = st.file_uploader("📤 Upload file PDF Audit (Tự động quét bảng thông số)", type="pdf")

if file_audit:
    with st.spinner("Hệ thống AI đang quét toàn bộ các bảng thông số..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        # Tìm mẫu trong kho
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            
            # Tính % tương đồng
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
            df_db['priority'] = df_db['customer_name'].apply(lambda x: 2 if "TP MỚI" in str(x).upper() else 1)
            df_db = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False])
            
            best = df_db.iloc[0]
            sim_percent = best['sim_score'] * 100

            # HIỂN THỊ HÌNH ẢNH
            st.subheader(f"🖼️ Mẫu khớp nhất: {best['file_name']} (Khớp: {sim_percent:.2f}%)")
            c_img1, c_img2 = st.columns(2)
            with c_img1: st.image(target['img'], caption="Bản Audit hiện tại", use_container_width=True)
            with c_img2: st.image(best['image_url'], caption=f"Mẫu gốc: {best['customer_name']}", use_container_width=True)

            # --- TỰ ĐỘNG KHỚP SIZE GIỮA 2 FILE ---
            st.divider()
            st.subheader("📊 Kết quả đối soát đa size")
            
            # Lấy danh sách Size có chung giữa 2 file
            common_sizes = set(target['all_specs'].keys()) & set(best['spec_json'].keys())
            
            if not common_sizes:
                st.warning("⚠️ Không tìm thấy size tương ứng để so sánh (Ví dụ: File gốc có size 10 nhưng file mới chỉ có S,M,L)")
            else:
                # Cho phép người dùng chọn size để xem chi tiết (mặc định chọn size đầu tiên khớp)
                selected_size = st.selectbox("Chọn Size để xem đối soát chi tiết:", list(common_sizes))
                
                audit_specs = target['all_specs'][selected_size]
                lib_specs = best['spec_json'][selected_size]
                
                res_table = []
                for pom, val_audit in audit_specs.items():
                    val_lib = lib_specs.get(pom, 0)
                    diff = val_audit - val_lib
                    status = "✅ Khớp" if abs(diff) < 0.1 else f"❌ Lệch ({diff:+.2f})"
                    res_table.append({
                        "Vị trí (POM)": pom, 
                        f"Gốc (Size {selected_size})": val_lib, 
                        f"Audit (Size {selected_size})": val_audit, 
                        "Kết quả": status
                    })
                
                st.table(pd.DataFrame(res_table))
        else: st.warning("Kho mẫu trống.")

