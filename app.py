import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.2", page_icon="📊")

# ================= INIT =================
@st.cache_resource
def init_supabase():
    try: return create_client(URL, KEY)
    except: return None

supabase = init_supabase()

# ================= AI MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= HELPER FUNCTIONS =================
def get_vector(img_bytes):
    try:
        if not img_bytes: return None
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
            return vec.tolist()
    except: return None

def extract_techpack(pdf_file):
    specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for tb in tables:
                    for row in tb:
                        if len(row) >= 2 and row[-1]:
                            desc = str(row[0]).strip().upper()
                            val = re.findall(r"\d+\.?\d*", str(row[-1]))
                            if val: specs[desc] = val[0]
        return {"spec": specs, "img": img_bytes}
    except: return None

def load_samples():
    try:
        res = supabase.table("ai_data").select("*").execute()
        return res.data if res.data else []
    except: return []

# ================= INTERFACE =================
samples = load_samples()

with st.sidebar:
    st.header("📦 Kho dữ liệu")
    st.metric("Số mẫu hiện có", len(samples))
    files = st.file_uploader("Upload Techpacks (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if files and st.button("🚀 Nạp vào hệ thống"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            data = extract_techpack(f)
            if data and data['img']:
                name = f.name.replace(".pdf","")
                img_path = f"{name}.png"
                
                # FIX LỖI UPSERT: Chuyển 'upsert' vào trong 'file_options'
                try:
                    supabase.storage.from_(BUCKET).upload(
                        path=img_path, 
                        file=data['img'], 
                        file_options={
                            "content-type": "image/png",
                            "x-upsert": "true" # Đây là cách ghi đè file mới
                        }
                    )
                    img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
                    vec = get_vector(data['img'])
                    
                    supabase.table("ai_data").upsert({
                        "file_name": name,
                        "vector": vec,
                        "spec_json": data['spec'],
                        "image_url": img_url
                    }).execute()
                except Exception as e:
                    st.error(f"Lỗi tại file {f.name}: {str(e)}")
            
            p_bar.progress((i + 1) / len(files))
        st.success("Xong!")
        st.rerun()

# MAIN
st.title("🔍 AI FASHION AUDITOR V34.2")
test_file = st.file_uploader("Kéo tệp PDF cần kiểm tra vào đây", type=["pdf"])

if test_file:
    test_data = extract_techpack(test_file)
    if test_data and test_data['img']:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(test_data['img'], caption="Mẫu kiểm tra", use_container_width=True)
            t_vec = get_vector(test_data['img'])
        
        if samples and t_vec:
            results = []
            for s in samples:
                sim = cosine_similarity([t_vec], [s['vector']])[0][0]
                results.append({"data": s, "sim": sim})
            
            best = max(results, key=lambda x: x['sim'])
            
            with col2:
                st.subheader(f"Kết quả AI: {round(best['sim']*100, 1)}%")
                st.progress(float(best['sim']))
                st.write(f"Khớp với: **{best['data']['file_name']}**")
                
                # So sánh bảng
                st.write("---")
                s_audit, s_db = test_data['spec'], best['data']['spec_json']
                comp = []
                for k in s_audit:
                    if k in s_db:
                        diff = round(float(s_audit[k]) - float(s_db[k]), 2)
                        comp.append({"Hạng mục": k, "Mẫu kiểm": s_audit[k], "Gốc": s_db[k], "Lệch": diff, "Kquả": "✅" if abs(diff)<0.2 else "❌"})
                if comp: st.table(pd.DataFrame(comp))
    else:
        st.error("Không trích xuất được dữ liệu!")
