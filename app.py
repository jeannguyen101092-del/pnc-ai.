import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
# Thay đổi URL và KEY theo thông tin Supabase của bạn
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase = create_client(URL, KEY)
except:
    st.error("Không thể kết nối Supabase. Vui lòng kiểm tra lại URL/KEY.")

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V94", page_icon="📏")

# CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .stTable { font-size: 13px !important; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    .main-header { font-size: 24px; font-weight: bold; color: #1E1E1E; margin-bottom: 20px; }
    div[data-testid="stExpander"] { border: 1px solid #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. MODEL AI & XỬ LÝ ẢNH =================
@st.cache_resource
def load_model():
    # Sử dụng ResNet18 để lấy vector đặc trưng (Feature Extraction)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

def get_vector(img_bytes):
    """Chuyển đổi hình ảnh thành vector để so sánh độ tương đồng"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return vec.astype(float).tolist()

# ================= 3. LOGIC TRÍCH XUẤT DỮ LIỆU PDF =================
def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "HOODIE", "ÁO", "COAT"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "VÁY", "ĐẦM"]): return "VÁY/ĐẦM"
    return "KHÁC"

def parse_val(t):
    """Xử lý các con số phức tạp như '1 1/2', '10.5', '12/1'"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        
        # Xử lý hỗn số '1 1/2'
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', txt)
        if mixed:
            g, n, d = mixed.groups()
            return float(g) + (float(n)/float(d))
            
        # Xử lý phân số '1/2'
        frac = re.match(r'(\d+)/(\d+)', txt)
        if frac:
            n, d = frac.groups()
            return float(n)/float(d)

        # Lấy số thập phân/số nguyên cuối cùng tìm thấy
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def is_pom_table(df):
    """Kiểm tra bảng có chứa từ khóa đo đạc kỹ thuật (POM) không"""
    keywords = ["WAIST", "CHEST", "HIP", "SLEEVE", "LENGTH", "SHOULDER", "THIGH", "RISE", "BUST", "NECK", "ARMHOLE"]
    text = " ".join(df.astype(str).values.flatten()).upper()
    score = sum(1 for k in keywords if k in text)
    return score >= 2 # Tìm thấy ít nhất 2 vị trí đo mới coi là bảng POM

def extract_pdf_v94(file, customer="Auto"):
    specs, img_bytes, category = {}, None, "KHÁC"
    try:
        file.seek(0)
        pdf_content = file.read()
        
        # 1. Dùng fitz để lấy ảnh trang đầu và text
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                category = detect_category(full_text, file.name)

        # 2. Dùng pdfplumber để trích xuất bảng số liệu
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1).fillna("")
                    if df.empty or not is_pom_table(df): continue
                    
                    n_col, v_col = -1, -1
                    # Tìm cột tên (Description) và cột giá trị (Spec/Sample)
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i
                            if any(x in v for x in ["NEW", "SAMPLE", "SPEC", "M", "32", "L"]): v_col = i
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARK"]): continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi PDF: {e}")
        return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO MẪU AI")
    try:
        res_db = supabase.table("ai_data").select("file_name", "category").execute()
        data_lib = res_db.data if res_db.data else []
        st.success(f"Đang lưu trữ: {len(data_lib)} mẫu")
    except: 
        st.warning("Chưa kết nối Database")
        data_lib = []

    new_files = st.file_uploader("Nạp Techpack mới (PDF)", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 NẠP DỮ LIỆU"):
        for f in new_files:
            with st.spinner(f"Đang phân tích {f.name}..."):
                data = extract_pdf_v94(f)
                if data and data['specs']:
                    vec = get_vector(data['img'])
                    # Upload ảnh lên Storage
                    path = f"img_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    # Lưu thông tin vào Table
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, 
                        "spec_json": data['specs'], "image_url": url, "category": data['category']
                    }).execute()
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT THÔNG MINH =================
st.markdown('<p class="main-header">🔍 AI SMART AUDITOR - V94</p>', unsafe_allow_html=True)

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Bản nhận được)", type="pdf")

if file_audit:
    with st.spinner("Đang trích xuất dữ liệu đối soát..."):
        target = extract_pdf_v94(file_audit)
    
    if target and target["specs"]:
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.image(target['img'], caption="Mẫu nhận diện từ PDF", use_container_width=True)
            st.metric("Loại hàng", target['category'])
            st.metric("Số điểm đo", len(target['specs']))
        
        with col_res2:
            # 1. Tìm kiếm trong kho mẫu cùng loại
            db_res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if not db_res.data:
                st.warning(f"Chưa có mẫu **{target['category']}** nào trong kho để đối chiếu.")
            else:
                # 2. So sánh Vector tìm mẫu giống nhất (Cosine Similarity)
                target_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                best_match = None
                max_sim = -1
                
                for item in db_res.data:
                    sim = cosine_similarity(target_vec, [item['vector']])[0][0]
                    if sim > max_sim:
                        max_sim = sim
                        best_match = item
                
                if best_match and max_sim > 0.7:
                    st.success(f"✅ Đã tìm thấy mẫu đối ứng: **{best_match['file_name']}** (Giống {max_sim:.1%})")
                    
                    # 3. Tạo bảng so sánh thông số
                    diff_list = []
                    master_specs = best_match['spec_json']
                    
                    # Lấy tất cả các tên điểm đo từ cả 2 bản
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
                    st.error("Không tìm thấy mẫu nào trong kho có hình ảnh tương đồng.")
    else:
        st.error("❌ Không tìm thấy bảng thông số kỹ thuật (POM) trong PDF này. Vui lòng kiểm tra lại định dạng file.")

st.divider()
st.caption("AI Fashion Auditor v94 - Hỗ trợ đối soát thông số tự động.")
