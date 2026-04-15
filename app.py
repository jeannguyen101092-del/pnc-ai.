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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor Pro", page_icon="🏢")

if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

st.markdown("""
    <style>
    .report-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    .stMetric { background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & UTILS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    try:
        if not t: return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        # Loại bỏ các đơn vị đo lường dính liền
        txt = re.sub(r'(cm|inch|in|mm|yds|gr)$', '', txt)
        if not txt or any(x in txt for x in ["date", "page", "total"]): return 0
        
        # Xử lý phân số (ví dụ 1 1/2 hoặc 1/2)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            parts = v_str.split()
            val = float(parts[0]) + eval(parts[1])
        elif '/' in v_str:
            val = eval(v_str)
        else:
            val = float(v_str)
        return val if val <= 300 else 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def extract_customer_name(text):
    patterns = [r"(?i)CUSTOMER[:\s]+([^\n]+)", r"(?i)BUYER[:\s]+([^\n]+)", r"(?i)KHÁCH HÀNG[:\s]+([^\n]+)"]
    for p in patterns:
        match = re.search(p, text)
        if match: return match.group(1).strip().upper()
    return "UNKNOWN"

# ================= 3. CORE: QUÉT ĐA SIZE & BẢNG THÔNG SỐ =================
def extract_pdf_multi_size(file):
    all_specs = {}
    img_bytes = None
    customer = "UNKNOWN"
    
    try:
        file.seek(0)
        pdf_content = file.read()
        # Lấy ảnh preview trang 1
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        full_text = " ".join([page.get_text() for page in doc])
        customer = extract_customer_name(full_text)
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    n_col = -1 # Cột chứa tên POM
                    size_cols = {} # Index cột : Tên Size
                    
                    # Quét 10 dòng đầu để tìm header
                    for r_idx in range(min(10, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        
                        # 1. Tìm cột POM
                        if n_col == -1:
                            for i, v in enumerate(row):
                                if any(x in v for x in ["DESCRIPTION", "POM", "MEASUREMENT", "POSITION"]):
                                    n_col = i; break
                        
                        # 2. Tìm các cột Size (S, M, L... hoặc số 30, 32...)
                        for i, v in enumerate(row):
                            if i == n_col or not v: continue
                            # Loại trừ cột Dung sai (Tolerance)
                            if any(tol in v for tol in ["TOL", "+/-", "DUNG SAI", "GRAD"]): continue
                            
                            is_size = (v.isdigit() or any(s == v for s in ["XXS","XS","S","M","L","XL","XXL","3XL"]))
                            if is_size and len(v) < 10:
                                size_cols[i] = v
                                
                        if n_col != -1 and size_cols: break
                    
                    # 3. Trích xuất dữ liệu nếu đã thấy Header
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0 and "SIZE" not in pom:
                                    all_specs[s_name][pom] = val
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except Exception as e:
        st.error(f"Lỗi quét PDF: {e}")
        return None

# ================= 4. GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.header("🏢 KHO MẪU HỆ THỐNG")
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng mẫu lưu trữ", f"{count_res.count or 0} Items")
    except: st.write("Đang kết nối...")
    
    st.divider()
    st.subheader("🚀 Nạp mẫu mới")
    new_files = st.file_uploader("Upload Techpacks", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data and data['all_specs']:
                vec = get_image_vector(data['img'])
                path = f"lib_{re.sub(r'\W+', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['all_specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": data['customer']
                }).execute()
        st.success("Đã nạp thành công!"); st.session_state.up_key += 1; st.rerun()

st.title("🔍 AI SMART AUDITOR - ĐỐI SOÁT THÔNG SỐ TỰ ĐỘNG")

file_audit = st.file_uploader("📤 Upload file PDF Audit (File cần kiểm tra)", type="pdf")

if file_audit:
    with st.spinner("Đang trích xuất dữ liệu Audit..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        # 1. Tìm mẫu khớp nhất trong kho bằng AI
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
            best = df_db.sort_values('sim_score', ascending=False).iloc[0]
            
            # 2. Hiển thị thông tin chung
            c1, c2 = st.columns(2)
            with c1: 
                st.image(target['img'], caption="Ảnh từ file Audit", use_container_width=True)
            with c2: 
                st.image(best['image_url'], caption=f"Mẫu gốc khớp {best['sim_score']*100:.1f}%", use_container_width=True)

            # 3. Đối soát Size
            st.divider()
            st.subheader("📊 Kết quả đối soát chi tiết")
            
            audit_sizes = list(target['all_specs'].keys())
            db_specs = best['spec_json']
            
            # Cho người dùng chọn size muốn xem đối soát
            sel_size = st.selectbox("Chọn Size để xem chi tiết:", audit_sizes)
            
            if sel_size in db_specs:
                spec_audit = target['all_specs'][sel_size]
                spec_ref = db_specs[sel_size]
                
                compare_data = []
                for pom, val_audit in spec_audit.items():
                    # Tìm POM tương ứng trong gốc (fuzzy match đơn giản bằng cách so chuỗi)
                    val_ref = spec_ref.get(pom, 0)
                    diff = val_audit - val_ref
                    status = "✅ KHỚP" if abs(diff) < 0.126 else f"❌ LỆCH ({diff:+.2f})"
                    compare_data.append({
                        "POM Name": pom,
                        "Bản Audit": val_audit,
                        "Mẫu Gốc": val_ref,
                        "Chênh lệch": diff,
                        "Trạng thái": status
                    })
                
                df_compare = pd.DataFrame(compare_data)
                
                # Highlight các dòng bị lệch
                def highlight_diff(row):
                    return ['background-color: #ffcccc' if "❌" in str(row['Trạng thái']) else '' for _ in row]
                
                st.table(df_compare.style.apply(highlight_diff, axis=1))
                
                # Thống kê nhanh
                total = len(df_compare)
                pass_count = len(df_compare[df_compare['Trạng thái'] == "✅ KHỚP"])
                st.metric("Tỷ lệ đạt chuẩn", f"{(pass_count/total)*100:.1f}%", f"{pass_count}/{total} POMs")
            else:
                st.warning(f"Size '{sel_size}' không tìm thấy trong dữ liệu mẫu gốc. Vui lòng kiểm tra lại bảng Size.")
    else:
        st.error("Không tìm thấy bảng thông số trong file PDF. Hãy đảm bảo PDF có bảng dạng cột Size rõ ràng.")

