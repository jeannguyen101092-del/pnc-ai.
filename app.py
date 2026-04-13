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

try:
    supabase = create_client(URL, KEY)
except:
    st.error("Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Smart Auditor V99", page_icon="📏")

# Khởi tạo key để xóa file upload sau khi nạp thành công
if 'upload_id' not in st.session_state: st.session_state.upload_id = 0

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return vec.tolist()

# ================= 3. UTILS NÂNG CẤP =================
def auto_detect_customer(text):
    t = str(text).upper()
    # THÊM VINEYARD VINES VÀ CÁC KHÁCH HÀNG KHÁC TẠI ĐÂY
    if "VINEYARD VINES" in t or "WHALE" in t: return "Vineyard Vines"
    if "REITMANS" in t: return "Reitmans"
    if "NIKE" in t: return "Nike"
    if "ADIDAS" in t: return "Adidas"
    return "Khác"

def extract_pdf_v99(file):
    specs, img_bytes, category, cust_found = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                # Nhận diện loại hàng mở rộng
                t_up = (full_text + file.name).upper()
                if any(x in t_up for x in ["PANT", "JEAN", "SKIRT", "QUẦN", "VÁY"]): category = "QUẦN/VÁY"
                elif any(x in t_up for x in ["SHIRT", "TOP", "TEE", "ÁO"]): category = "ÁO"
                cust_found = auto_detect_customer(full_text)
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if sum(1 for k in ["WAIST", "CHEST", "LENGTH", "HIP", "BUST"] if k in " ".join(df.astype(str).values.flatten()).upper()) < 2: continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i
                            if any(x in v for x in ["NEW", "SAMPLE", "SPEC", "M", "32", "S"]): v_col = i
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or "TOL" in name: continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": cust_found}
    except: return None

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO MASTER")
    
    # HIỂN THỊ SỐ LƯỢNG MẪU TRONG KHO
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        total_items = count_res.count if count_res.count else 0
        st.info(f"📊 Tổng cộng trong kho: **{total_items}** mẫu")
    except:
        st.warning("Chưa có dữ liệu trong kho.")

    st.divider()
    new_files = st.file_uploader("Upload Master PDF", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.upload_id}")
    
    if new_files and st.button("🚀 NẠP DỮ LIỆU"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, f in enumerate(new_files):
            # Cập nhật % tiến độ
            percent = int((i + 1) / len(new_files) * 100)
            progress_bar.progress(percent)
            status_text.text(f"Đang nạp {percent}%: {f.name}")
            
            data = extract_pdf_v99(f)
            if data and data['specs']:
                # Upload ảnh
                path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type":"image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                # Insert DB
                supabase.table("ai_data").insert({
                    "file_name": f.name, 
                    "customer": data['customer'], 
                    "prod_id": f.name.split('.')[0], # Lấy mã trước dấu chấm
                    "vector": get_vector(data['img']), 
                    "spec_json": data['specs'], 
                    "image_url": url, 
                    "category": data['category']
                }).execute()
        
        status_text.success("🎉 Đã hoàn thành nạp 100% dữ liệu!")
        st.session_state.upload_id += 1
        st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V99")

c_f1, c_f2 = st.columns(2)
with c_f1: 
    sel_cust = st.selectbox("🎯 Lọc Khách hàng:", ["Tất cả", "Vineyard Vines", "Reitmans", "Nike", "Adidas"])
with c_f2: 
    sel_prod = st.text_input("🆔 Tìm mã Style (Production):").strip().upper()

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    target = extract_pdf_v99(file_audit)
    if target:
        st.info(f"✨ Hệ thống nhận diện: Khách hàng **{target['customer']}** | Loại hàng **{target['category']}**")
        
        # Logic tìm kiếm
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        if sel_prod: query = query.ilike("prod_id", f"%{sel_prod}%")
        db_res = query.execute()
        
        if db_res.data:
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match, max_sim = None, -1.0
            for item in db_res.data:
                sim = float(cosine_similarity(t_vec, [item['vector']]))
                if sim > max_sim:
                    max_sim = sim; best_match = item

            if best_match:
                st.subheader(f"📊 Kết quả đối soát (Độ tương đồng: {max_sim:.1%})")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="BẢN AUDIT", use_container_width=True)
                with c2: st.image(best_match['image_url'], caption=f"BẢN MASTER: {best_match['file_name']}", use_container_width=True)
                
                # Hiển thị thông số (Bảng so sánh đã có trong logic của bạn)
                # ... phần vẽ bảng thông số giữ nguyên ...
