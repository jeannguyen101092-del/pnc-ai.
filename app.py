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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96", page_icon="📏")

# CSS
st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .img-box { border: 2px solid #e6e9ef; border-radius: 8px; padding: 10px; background: #fff; text-align: center; }
    .highlight { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return vec.tolist()

# ================= 3. UTILS =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg"]): return 0
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', txt)
        if mixed:
            g, n, d = mixed.groups()
            return float(g) + (float(n)/float(d))
        frac = re.match(r'(\d+)/(\d+)', txt)
        if frac:
            n, d = frac.groups()
            return float(n)/float(d)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "ÁO"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "VÁY", "ĐẦM"]): return "VÁY/ĐẦM"
    return "KHÁC"

def extract_pdf_v96(file):
    specs, img_bytes, category = {}, None, "KHÁC"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                category = detect_category("".join([p.get_text() for p in doc]), file.name)
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    # Kiểm tra bảng POM
                    if sum(1 for k in ["WAIST", "CHEST", "LENGTH", "HIP", "SLEEVE"] if k in " ".join(df.astype(str).values.flatten()).upper()) < 2: continue
                    
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
        return {"specs": specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi PDF: {e}")
        return None

# ================= 4. SIDEBAR (NẠP DỮ LIỆU) =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MASTER")
    inp_cust = st.selectbox("Khách hàng:", ["Reitmans", "Nike", "Adidas", "Khác"])
    inp_prod = st.text_input("Mã Production/Style:", placeholder="F25-XXX")
    
    new_files = st.file_uploader("Upload Techpack Master", type="pdf", accept_multiple_files=True)
    if new_files and st.button("🚀 NẠP VÀO HỆ THỐNG"):
        for f in new_files:
            data = extract_pdf_v96(f)
            if data and data['specs']:
                # 1. Xử lý Ảnh
                path = f"master_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                # 2. Xử lý Vector
                vec = get_vector(data['img'])
                
                # 3. Gửi lên DB (Có bẫy lỗi chi tiết)
                try:
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "customer": inp_cust, "prod_id": inp_prod,
                        "vector": vec, "spec_json": data['specs'], "image_url": img_url, 
                        "category": data['category']
                    }).execute()
                    st.success(f"Đã nạp: {f.name}")
                except Exception as db_err:
                    st.error(f"Lỗi DB với file {f.name}: {db_err}")

# ================= 5. MAIN (ĐỐI SOÁT) =================
st.title("🔍 AI SMART AUDITOR - V96")

# Bộ lọc
c_f1, c_f2 = st.columns(2)
with c_f1:
    sel_cust = st.selectbox("🎯 Lọc theo Khách hàng:", ["Tất cả", "Reitmans", "Nike", "Adidas"])
with c_f2:
    sel_prod = st.text_input("🆔 Tìm theo mã Style/Prod (Để trống nếu tìm tự động):").strip().upper()

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Bản Audit)", type="pdf")

if file_audit:
    target = extract_pdf_v96(file_audit)
    if target and target["specs"]:
        # TRUY VẤN
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        if sel_prod: query = query.ilike("prod_id", f"%{sel_prod}%")
        
        db_res = query.execute()
        
        if not db_res.data:
            st.warning("⚠️ Không tìm thấy mẫu phù hợp trong kho.")
        else:
            # So sánh AI
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match = None
            max_sim = -1
            
            for item in db_res.data:
                sim = cosine_similarity(t_vec, [item['vector']])[0][0]
                if sim > max_sim:
                    max_sim = sim; best_match = item

            if best_match:
                # HIỂN THỊ SONG SONG
                st.subheader(f"📊 Kết quả đối soát (Độ giống: {max_sim:.1%})")
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown('<div class="img-box">', unsafe_allow_html=True)
                    st.image(target['img'], caption="BẢN MỚI (AUDIT)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_i2:
                    st.markdown('<div class="img-box">', unsafe_allow_html=True)
                    st.image(best_match['image_url'], caption=f"BẢN GỐC (MASTER): {best_match['file_name']}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # SO SÁNH THÔNG SỐ
                st.divider()
                st.markdown(f"📦 **Production ID:** {best_match['prod_id']} | **Customer:** {best_match['customer']}")
                
                m_specs = best_match['spec_json']
                all_poms = sorted(set(list(target['specs'].keys()) + list(m_specs.keys())))
                
                rows = []
                for pom in all_poms:
                    v_m, v_t = m_specs.get(pom, 0), target['specs'].get(pom, 0)
                    diff = round(v_t - v_m, 3)
                    res = "✅ Khớp" if abs(diff) < 0.125 else f"❌ Lệch ({diff})"
                    rows.append({"Vị trí đo": pom, "Master": v_m, "Audit": v_t, "Chênh lệch": res})
                
                st.table(pd.DataFrame(rows))
    else:
        st.error("Không tìm thấy bảng POM trong PDF.")

st.caption("AI Smart Auditor V96 - Kiểm soát chất lượng thông số tự động.")
