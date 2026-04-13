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
except Exception as e:
    st.error(f"Kết nối thất bại: {e}")

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V97", page_icon="📏")

# CSS
st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .img-box { border: 2px solid #e6e9ef; border-radius: 8px; padding: 10px; background: #fff; text-align: center; }
    .highlight { background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid #28a745; }
    </style>
    """, unsafe_allow_html=True)

# Khởi tạo key để xóa file upload sau khi nạp
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

# ================= 3. TRÍCH XUẤT THÔNG MINH =================
def auto_detect_customer(text):
    """Tự động nhận diện khách hàng từ nội dung PDF"""
    t = str(text).upper()
    if "REITMANS" in t: return "Reitmans"
    if "NIKE" in t: return "Nike"
    if "ADIDAS" in t: return "Adidas"
    if "PUMA" in t: return "Puma"
    return "Khác"

def extract_pdf_v97(file):
    specs, img_bytes, category, cust_found = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                category = detect_category(full_text, file.name)
                cust_found = auto_detect_customer(full_text) # QUÉT TÊN KHÁCH HÀNG
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    text_blob = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "LENGTH", "HIP"] if k in text_blob) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(8).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i
                            if any(x in v for x in ["NEW", "SAMPLE", "SPEC", "M", "32"]): v_col = i
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or "TOL" in name: continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": cust_found}
    except Exception as e:
        st.error(f"Lỗi PDF: {e}")
        return None

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["PANT", "JEAN", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "ÁO"]): return "ÁO"
    return "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        nums = re.findall(r"[-+]?\d*\.\d+|\d+/\d+|\d+", txt)
        if not nums: return 0
        v = nums[0]
        if '/' in v:
            p = v.split('/')
            return float(p[0])/float(p[1])
        return float(v)
    except: return 0

# ================= 4. SIDEBAR - NẠP KHO & TỰ XÓA =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MASTER")
    # Reset file uploader bằng cách đổi Key
    new_files = st.file_uploader("Upload Techpack Master", type="pdf", 
                                 accept_multiple_files=True, 
                                 key=f"uploader_{st.session_state.upload_id}")
    
    if new_files and st.button("🚀 NẠP DỮ LIỆU & TỰ XÓA FILE"):
        success_count = 0
        for f in new_files:
            with st.spinner(f"Đang xử lý {f.name}..."):
                data = extract_pdf_v97(f)
                if data and data['specs']:
                    # 1. Upload ảnh
                    path = f"master_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    # 2. Insert DB - Tự động lấy Customer & Style
                    style_code = f.name.split('.')[0] # Lấy tên file làm Style
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "customer": data['customer'], # TỰ ĐỘNG QUÉT ĐƯỢC
                        "prod_id": style_code, 
                        "vector": get_vector(data['img']),
                        "spec_json": data['specs'], 
                        "image_url": img_url, 
                        "category": data['category']
                    }).execute()
                    success_count += 1
        
        if success_count > 0:
            st.success(f"Đã nạp {success_count} file thành công!")
            # Tăng ID để xóa danh sách file đã upload trên giao diện
            st.session_state.upload_id += 1 
            st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V97")

col_f1, col_f2 = st.columns(2)
with col_f1:
    sel_cust = st.selectbox("🎯 Lọc theo Khách hàng:", ["Tất cả", "Reitmans", "Nike", "Adidas", "Puma"])
with col_f2:
    sel_prod = st.text_input("🆔 Tìm mã Style (Nếu cần lọc nhanh):").strip().upper()

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    target = extract_pdf_v97(file_audit)
    if target:
        st.info(f"✨ Hệ thống nhận diện: Khách hàng **{target['customer']}** | Loại hàng **{target['category']}**")
        
        # Truy vấn tìm mẫu Master tương ứng
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        if sel_prod: query = query.ilike("prod_id", f"%{sel_prod}%")
        
        db_res = query.execute()
        
        if db_res.data:
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match = None; max_sim = -1
            for item in db_res.data:
                sim = cosine_similarity(t_vec, [item['vector']])
                if sim > max_sim: max_sim = sim; best_match = item

            if best_match:
                st.subheader(f"📊 Kết quả đối soát (Độ tương đồng: {max_sim:.1%})")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="BẢN MỚI (AUDIT)", use_container_width=True)
                with c2: st.image(best_match['image_url'], caption=f"BẢN GỐC (MASTER): {best_match['file_name']}", use_container_width=True)

                # Bảng thông số
                m_specs = best_match['spec_json']
                all_poms = sorted(set(list(target['specs'].keys()) + list(m_specs.keys())))
                rows = []
                for pom in all_poms:
                    v_m, v_t = m_specs.get(pom, 0), target['specs'].get(pom, 0)
                    diff = round(v_t - v_m, 3)
                    res = "✅ Khớp" if abs(diff) < 0.125 else f"❌ Lệch ({diff})"
                    rows.append({"Điểm đo": pom, "Master": v_m, "Audit": v_t, "Kết quả": res})
                st.table(pd.DataFrame(rows))
