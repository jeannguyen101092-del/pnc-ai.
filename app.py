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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V106", page_icon="🏢")

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
    """Xử lý thông số đo (bao gồm cả phân số Denim)"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 10 or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs", "date"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            parts = v_str.split()
            return float(parts[0]) + (eval(parts[1]) if '/' in parts[1] else 0)
        return eval(v_str) if '/' in v_str else float(v_str)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE (DENIM OPTIMIZED) =================
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
                    n_col = -1; size_cols = {}
                    
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            # Mở rộng nhận diện cột mô tả cho Denim (SPECIFICATION)
                            if any(x in v for x in ["DESCRIPTION", "POM", "POSITION", "SPECIFICATION"]): 
                                n_col = i
                            # Nhận diện size số hoặc chữ
                            elif v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "XXL", "XS"]):
                                if i != n_col: size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0 and pom not in ["DESCRIPTION", "SPECIFICATION", "POM"]:
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except Exception as e: return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{count_res.count or 0} file")
    except: pass
    
    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới vào kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data and data['all_specs']:
                vec = get_image_vector(data['img'])
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": data['customer']
                }).execute()
        st.success("Nạp thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. ĐỐI SOÁT TỰ ĐỘNG =================
st.title("🔍 AI SMART AUDITOR V106")

file_audit = st.file_uploader("📤 Upload file PDF Audit", type="pdf")

if file_audit:
    with st.spinner("Hệ thống AI đang quét bảng thông số..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
            df_db = df_db.sort_values(by='sim_score', ascending=False)
            
            best = df_db.iloc[0]
            sim_percent = best['sim_score'] * 100

            st.subheader(f"🖼️ Mẫu khớp nhất: {best['file_name']} (Khớp: {sim_percent:.2f}%)")
            c1, c2 = st.columns(2)
            with c1: st.image(target['img'], caption="Bản Audit", use_container_width=True)
            with c2: st.image(best['image_url'], caption=f"Mẫu gốc: {best['customer_name']}", use_container_width=True)

            st.divider()
            # HIỂN THỊ BẢNG THÔNG SỐ (FIXED)
            common_sizes = sorted(list(set(target['all_specs'].keys()) & set(best['spec_json'].keys())))
            
            if common_sizes:
                sel_size = st.selectbox("Chọn Size đối soát:", common_sizes)
                aud_data = target['all_specs'][sel_size]
                lib_data = best['spec_json'][sel_size]
                
                res_table = []
                for pom, val_aud in aud_data.items():
                    val_lib = lib_data.get(pom, 0)
                    diff = round(val_aud - val_lib, 3)
                    res_table.append({
                        "Vị trí đo (POM)": pom, 
                        "Mẫu Gốc": val_lib, 
                        "Bản Audit": val_aud, 
                        "Lệch": diff,
                        "Kết quả": "✅ Khớp" if abs(diff) < 0.1 else "❌ Lệch"
                    })
                st.table(pd.DataFrame(res_table))
            else:
                # Nếu không khớp tên size (VD: S/M/L vs 24/25/26), hiện size của bản Audit để người dùng xem
                st.warning("⚠️ Không khớp tên Size. Đang hiển thị thông số bản Audit:")
                audit_size_list = list(target['all_specs'].keys())
                sel_size = st.selectbox("Chọn Size bản Audit:", audit_size_list)
                st.table(pd.DataFrame([{"POM": k, "Giá trị": v} for k, v in target['all_specs'][sel_size].items()]))
