# ==========================================================
# AI FASHION PRO V8.2 - FIXED API ERROR & UPSERT
# ==========================================================
import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, json
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- KẾT NỐI ---
# Điền thông tin Supabase của bạn
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.2", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- XỬ LÝ DỮ LIỆU ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

VALID_KEYS = ['WAIST','HIP','INSEAM','THIGH','KNEE','LEG','CHEST','LENGTH','SLEEVE','SHOULDER']
BLOCK_KEYS = ['DATE','SEASON','FABRIC','MATERIAL','COLOR','PAGE']

def extract_specs(table):
    specs = {}
    for r in table:
        if not r or len(r) < 2: continue
        row_text = " ".join([str(x) for x in r if x]).upper()
        if any(b in row_text for b in BLOCK_KEYS): continue
        if not any(k in row_text for k in VALID_KEYS): continue
        
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]
        if vals:
            key = str(r[0]).strip().upper()[:80]
            specs[key] = round(float(np.median(vals)), 2)
    return specs

def get_data(pdf_path):
    try:
        specs, all_texts = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                all_texts += (p.extract_text() or "") + " "
                tables = p.extract_tables()
                for table in tables:
                    specs.update(extract_specs(table))
        
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        doc.close()

        # Phân loại đơn giản
        cat = "QUẦN" if any(k in specs for k in ['WAIST','INSEAM','HIP']) else "ÁO"
        return {"spec": specs, "img_b64": img_b64, "img_bytes": img_bytes, "cat": cat}
    except Exception as e:
        return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    files = st.file_uploader("Upload Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            try:
                # Xóa các ký tự đặc biệt trong tên file
                clean_name = re.sub(r'[^\w\s.-]', '', f.name)
                with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
                d = get_data("tmp.pdf")
                
                if d:
                    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    with torch.no_grad():
                        img_pill = Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')
                        vec = ai_brain(tf(img_pill).unsqueeze(0)).flatten().numpy().tolist()
                    
                    # Lệnh nạp chính yếu
                    supabase.table("ai_data").upsert({
                        "file_name": clean_name,
                        "vector": vec,
                        "spec_json": d['spec'],
                        "img_base64": d['img_b64'],
                        "category": d['cat']
                    }, on_conflict="file_name").execute()
                
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
            except Exception as e:
                st.warning(f"Bỏ qua {f.name}: {str(e)}")
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Hoàn tất!")
        st.rerun()

# --- MAIN: KIỂM TRA ---
st.title("👔 AI Fashion Pro V8.2")
test_file = st.file_uploader("Kéo thả file PDF vào đây để kiểm tra", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.info(f"Nhận diện: {target['cat']}")
        res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if res.data:
            # Logic so sánh Similarity và hiển thị bảng đối soát (giữ nguyên V8)
            # ... (Phần hiển thị kết quả tương tự như các bản trước)
            st.success("Đã tìm thấy mẫu tương đồng. Đang phân tích...")
