import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client

# --- CONFIG (Điền URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V46.6", page_icon="🔍")

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
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_name(t):
    """Chuẩn hóa tên để soi đúng vị trí dòng"""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

def get_vec(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        return model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()

# --- TRÍCH XUẤT ---
def extract_data(pdf_file):
    specs, img_bytes = {}, None
    pdf_content = pdf_file.read()
    doc = fitz.open(stream=io.BytesIO(pdf_content))
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(0.8, 0.8)).tobytes("jpg", jpg_quality=50)
    
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb).fillna("")
                if len(df.columns) < 2: continue
                for i, row in df.iterrows():
                    name = str(row[0]).strip().upper()
                    val = parse_val(row[1])
                    if len(name) > 3 and val > 0:
                        specs[name] = float(val)
    return {"specs": specs, "img": img_bytes}

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 KHO MẪU")
    files = st.file_uploader("Nạp mẫu chuẩn", accept_multiple_files=True)
    if files and st.button("🚀 NẠP HỆ THỐNG"):
        for f in files:
            d = extract_data(f)
            if d['specs']:
                vec = get_vec(d['img'])
                path = f"lib/{f.name}.jpg"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'], "image_url": img_url
                }).execute()
        st.success("Xong!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V46.6")
f_test = st.file_uploader("Upload file kiểm", type="pdf")

if f_test:
    target = extract_data(f_test)
    db = supabase.table("ai_data").select("*").execute()
    
    if db.data and target['specs']:
        v_test = np.array(get_vec(target['img']))
        matches = []
        for row in db.data:
            v_ref = np.array(row['vector'])
            sim = np.dot(v_test, v_ref) / (np.linalg.norm(v_test) * np.linalg.norm(v_ref))
            matches.append({"row": row, "sim": float(sim)})
        
        # Lấy mẫu khớp nhất
        best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
        m_data = best['row']
        
        st.subheader(f"✨ Khớp mẫu: {m_data['file_name']} ({best['sim']*100:.1f}%)")
        
        # 1. HIỂN THỊ HÌNH ẢNH ĐỐI CHIẾU
        c1, c2 = st.columns(2)
        with c1: st.image(target['img'], caption="FILE KIỂM")
        with c2: st.image(m_data['image_url'], caption="FILE MẪU GỐC")

        # 2. HIỂN THỊ THÔNG SỐ ĐỐI CHIẾU ĐÚNG DÒNG
        st.write("### 📊 Bảng so khớp thông số")
        ref_specs = m_data['spec_json']
        # Map tên chuẩn hóa -> giá trị để tra cứu
        ref_map = {clean_name(k): (k, v) for k, v in ref_specs.items()}
        
        diff_list = []
        for p_name, v_new in target['specs'].items():
            p_key = clean_name(p_name)
            # Tìm xem trong file mẫu có vị trí này không
            orig_name, v_ref = ref_map.get(p_key, (p_name, 0))
            diff = round(v_new - v_ref, 3)
            
            diff_list.append({
                "Vị trí đo (POM)": p_name,
                "Thực tế": v_new,
                "Mẫu gốc": v_ref if v_ref > 0 else "N/A",
                "Lệch": diff,
                "Kết quả": "✅" if abs(diff) <= 0.125 else "❌"
            })
        
        st.table(pd.DataFrame(diff_list))

if st.button("RESET"): st.rerun()
