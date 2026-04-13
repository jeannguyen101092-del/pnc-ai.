import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH & KẾT NỐI =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase = create_client(URL, KEY)
except:
    st.error("Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V95", page_icon="📏")

# Giao diện CSS
st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    .img-container { border: 2px solid #f0f2f6; border-radius: 10px; padding: 5px; background: white; }
    .metric-box { padding: 10px; background: #f8f9fa; border-left: 5px solid #007bff; border-radius: 5px; }
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
    return vec.astype(float).tolist()

# ================= 3. UTILS & PARSER =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
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

def extract_pdf_v95(file):
    specs, img_bytes, category = {}, None, "KHÁC"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                category = detect_category(full_text, file.name)
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1).fillna("")
                    text_blob = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "HIP", "SLEEVE", "LENGTH"] if k in text_blob) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
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
        st.error(f"Lỗi: {e}")
        return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 NẠP KHO MẪU MASTER")
    c_customer = st.selectbox("Khách hàng:", ["Reitmans", "Nike", "Adidas", "Khác"])
    c_prod = st.text_input("Mã Production/Style:", placeholder="Ví dụ: F25R09-480")
    
    new_files = st.file_uploader("Chọn file PDF Master", type="pdf", accept_multiple_files=True)
    if new_files and st.button("🚀 LƯU VÀO KHO AI"):
        for f in new_files:
            data = extract_pdf_v95(f)
            if data and data['specs']:
                vec = get_vector(data['img'])
                path = f"master_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "customer": c_customer, "prod_id": c_prod,
                    "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category']
                }).execute()
        st.success("Đã nạp kho thành công!")
        st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V95")

# Bộ lọc đối soát
col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 2])
with col_f1:
    filter_cust = st.selectbox("🎯 Đối soát Khách hàng:", ["Tất cả", "Reitmans", "Nike", "Adidas"])
with col_f2:
    filter_prod = st.text_input("🆔 Tìm theo mã Production (Style):").strip().upper()

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Bản Audit)", type="pdf")

if file_audit:
    target = extract_pdf_v95(file_audit)
    if target and target["specs"]:
        # TRUY VẤN DỮ LIỆU CÓ ĐIỀU KIỆN
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if filter_cust != "Tất cả": query = query.eq("customer", filter_cust)
        if filter_prod: query = query.ilike("prod_id", f"%{filter_prod}%")
        
        db_res = query.execute()
        
        if not db_res.data:
            st.warning(f"Không tìm thấy mẫu phù hợp trong kho cho điều kiện: {filter_cust} | {filter_prod}")
        else:
            # So sánh AI tìm mẫu giống nhất
            target_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match = None
            max_sim = -1
            
            for item in db_res.data:
                sim = cosine_similarity(target_vec, [item['vector']])[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = item

            if best_match:
                # HIỂN THỊ HÌNH ẢNH SONG SONG
                st.subheader(f"📊 Kết quả đối soát (Độ tương đồng AI: {max_sim:.1%})")
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown('<div class="img-container">', unsafe_allow_html=True)
                    st.image(target['img'], caption="BẢN ĐANG KIỂM TRA (AUDIT)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_img2:
                    st.markdown('<div class="img-container">', unsafe_allow_html=True)
                    st.image(best_match['image_url'], caption=f"BẢN GỐC TRONG KHO (MASTER): {best_match['file_name']}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # BẢNG SO SÁNH THÔNG SỐ
                st.divider()
                st.markdown(f"**Khách hàng:** {best_match.get('customer')} | **Mã Prod:** {best_match.get('prod_id')}")
                
                diff_list = []
                master_specs = best_match['spec_json']
                all_names = sorted(set(list(target['specs'].keys()) + list(master_specs.keys())))
                
                for name in all_names:
                    v_master = master_specs.get(name, 0)
                    v_target = target['specs'].get(name, 0)
                    diff = round(v_target - v_master, 3)
                    status = "✅ Khớp" if abs(diff) < 0.1 else f"❌ Lệch ({diff})"
                    diff_list.append({
                        "Điểm đo (POM)": name,
                        "Gốc (Master)": v_master,
                        "Mới (Audit)": v_target,
                        "Kết quả": status
                    })
                
                st.table(pd.DataFrame(diff_list))
    else:
        st.error("Không thể đọc được bảng thông số từ file PDF này.")

st.divider()
st.caption("AI Smart Auditor V95 - Tự động hóa đối soát ngành may.")
