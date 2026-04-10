import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V46.8", page_icon="🔍")

# --- MODEL AI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- UTILS ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_name(t):
    """Chuẩn hóa tên để soi đúng vị trí dòng thông số"""
    if not t: return ""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

def get_vec(img_bytes):
    """Trích xuất vector ảnh chuẩn float64"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        return model_ai(tf(img).unsqueeze(0)).flatten().numpy().astype('float64').tolist()

# --- TRÍCH XUẤT ---
def extract_data(pdf_file):
    specs, img_bytes = {}, None
    pdf_content = pdf_file.read()
    doc = fitz.open(stream=io.BytesIO(pdf_content))
    if len(doc) > 0:
        # Ảnh JPG 50% siêu nhẹ để đối chiếu 2 bên nhanh
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(0.8, 0.8)).tobytes("jpg", jpg_quality=50)
    
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb).fillna("")
                if len(df.columns) < 2: continue
                for _, row in df.iterrows():
                    name = str(row[0]).strip().upper()
                    val = parse_val(row[1])
                    if len(name) > 3 and val > 0:
                        specs[name] = float(val)
    return {"specs": specs, "img": img_bytes}

# --- SIDEBAR: KHO MẪU ---
with st.sidebar:
    st.header("📂 KHO MẪU")
    try:
        res_check = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Số mẫu hiện có", res_check.count if res_check.count else 0)
    except: st.error("⚠️ Lỗi kết nối bảng 'ai_data'")

    files = st.file_uploader("Nạp mẫu chuẩn", accept_multiple_files=True)
    if files and st.button("🚀 NẠP HỆ THỐNG"):
        for f in files:
            d = extract_data(f)
            if d['specs']:
                vec = get_vec(d['img'])
                path = f"lib/{f.name.replace('.pdf', '.jpg')}"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, 
                    "spec_json": d['specs'], "image_url": img_url
                }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V46.8 PRO")
f_test = st.file_uploader("Upload file cần đối soát", type="pdf")

if f_test:
    target = extract_data(f_test)
    db = supabase.table("ai_data").select("*").execute()
    
    if db.data and target['specs']:
        # 1. TÌM KIẾM MẪU BẰNG AI (ÉP KIỂU AN TOÀN)
        v_test = np.array(get_vec(target['img']), dtype='float64')
        matches = []
        for row in db.data:
            v_raw = row.get('vector')
            # 🔥 FIX LỖI VALUEERROR: Kiểm tra định dạng vector từ DB
            if v_raw and isinstance(v_raw, list) and len(v_raw) == 512:
                try:
                    v_ref = np.array(v_raw, dtype='float64')
                    denom = (np.linalg.norm(v_test) * np.linalg.norm(v_ref))
                    sim = np.dot(v_test, v_ref) / denom if denom != 0 else 0
                    matches.append({"row": row, "sim": float(sim)})
                except: continue # Bỏ qua nếu dòng dữ liệu lỗi
        
        if matches:
            best_list = sorted(matches, key=lambda x: x['sim'], reverse=True)
            m_data = best_list[0]['row']
            
            st.subheader(f"🏆 Khớp với mẫu: {m_data['file_name']} ({best_list[0]['sim']*100:.1f}%)")
            
            # 2. HIỂN THỊ ẢNH ĐỐI CHIẾU 2 BÊN
            col_im1, col_im2 = st.columns(2)
            with col_im1: st.image(target['img'], caption="BẢN VẼ ĐANG KIỂM", use_container_width=True)
            with col_im2: st.image(m_data['image_url'], caption="MẪU GỐC TRONG KHO", use_container_width=True)

            # 3. HIỂN THỊ BẢNG ĐỐI CHIẾU ĐÚNG DÒNG
            st.write("### 📊 Chi tiết đối soát thông số")
            ref_specs = m_data['spec_json']
            ref_map = {clean_name(k): v for k, v in ref_specs.items()}
            
            results = []
            for p_name, v_new in target['specs'].items():
                p_key = clean_name(p_name)
                v_ref = ref_map.get(p_key, 0) # Tìm đúng vị trí theo tên POM
                diff = round(v_new - v_ref, 3)
                
                results.append({
                    "Vị trí đo (POM)": p_name,
                    "Thực tế": v_new,
                    "Mẫu Gốc": v_ref if v_ref > 0 else "N/A",
                    "Chênh lệch": diff,
                    "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })
            st.table(pd.DataFrame(results))
        else:
            st.error("Không tìm thấy mẫu tương đồng hợp lệ trong kho.")

if st.button("RESET"): st.rerun()
