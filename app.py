import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (CHỈ CẦN URL VÀ KEY SUPABASE) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs" # Tên bucket bạn vừa tạo trên Supabase

# Khởi tạo Supabase
try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Lỗi cấu hình Supabase: {e}")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11", page_icon="👔")

# ================= AI ENGINE (RESNET18) =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM NÉN ẢNH SANG WEBP SIÊU NHẸ =================
def compress_to_webp(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    # Nén WebP chất lượng 70 để cực nhẹ nhưng vẫn đủ nét cho AI
    img.save(buf, format="WEBP", quality=70, method=6)
    return buf.getvalue()

# ================= HÀM NẠP ẢNH VÀO SUPABASE STORAGE (THAY GITHUB) =================
def upload_to_storage(img_bytes, filename):
    try:
        # Làm sạch tên file và thêm đuôi .webp
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename) + ".webp"
        
        # Đẩy ảnh lên kho lưu trữ Supabase (ghi đè nếu trùng tên)
        supabase.storage.from_(BUCKET_NAME).upload(
            path=clean_name,
            file=img_bytes,
            file_options={"content-type": "image/webp", "upsert": "true"}
        )
        
        # Lấy link ảnh Public chuẩn
        res = supabase.storage.from_(BUCKET_NAME).get_public_url(clean_name)
        return res
    except Exception as e:
        st.error(f"❌ Lỗi kho ảnh Supabase: {e}")
        return None

# ================= TRÍCH XUẤT & PHÂN LOẠI CHUẨN =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def classify_logic(specs, text, name):
    txt = (text + name).upper()
    inseam, length = 0, 0
    for k, v in specs.items():
        if 'INSEAM' in k: inseam = max(inseam, v)
        if 'LENGTH' in k: length = max(length, v)

    if 'CARGO' in txt: return "QUẦN CARGO"
    # Logic phân biệt Quần dài vs Short chuẩn xác
    if inseam >= 22 or length >= 30: return "QUẦN DÀI"
    if 0 < inseam <= 15 or 0 < length <= 22: return "QUẦN SHORT"
    if any(k in txt for k in ['SHIRT', 'SƠ MI']): return "ÁO SƠ MI"
    if 'DRESS' in txt: return "ĐẦM"
    return "ÁO"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    for r in tb:
                        if not r: continue
                        txt_r = " | ".join([str(x) for x in r if x]).upper()
                        if any(k in txt_r for k in ['INSEAM','WAIST','HIP','LENGTH','CHEST']):
                            vals = [parse_val(x) for x in r if x and parse_val(x) > 0]
                            if vals: specs[txt_r[:100]] = round(float(np.median(vals)), 2)
        
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img_bytes, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= GIAO DIỆN CHÍNH =================
st.title("👔 AI Fashion Pro V11 (Supabase Storage Edition)")

with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    files = st.file_uploader("Upload PDF nạp kho", accept_multiple_files=True)
    
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_box = st.empty()
        for f in files:
            p_box.info(f"🔄 Đang xử lý: {f.name}...")
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            
            if d:
                # 1. Nén sang WebP siêu nhẹ
                img_webp = compress_to_webp(d['img'])
                
                # 2. Nạp thẳng vào Supabase Storage
                img_url = upload_to_storage(img_webp, f.name)
                
                if img_url:
                    # 3. AI Vector
                    tf = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        img_pil = Image.open(io.BytesIO(img_webp))
                        vec = ai_brain(tf(img_pil).unsqueeze(0)).flatten().numpy().tolist()

                    # 4. Lưu vào bảng Database
                    supabase.table("ai_data").upsert({
                        "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                        "img_url": img_url, "category": d['cat']
                    }, on_conflict="file_name").execute()
                    st.toast(f"✅ Thành công: {f.name}")
        
        st.success("🏁 Đã nạp kho xong!")
        if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.rerun()

# ================= PHẦN TEST & SO SÁNH =================
test_file = st.file_uploader("Tải file Test đối chứng (PDF)", type="pdf")
if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.divider()
        col1, col2 = st.columns()
        with col1:
            st.image(target['img'], caption=f"Loại: {target['cat']}")
        with col2:
            st.write("### Thông số quét được:")
            st.json(target['spec'])

        # Tìm cùng loại trong kho
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if db.data:
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()

            results = []
            for item in db.data:
                if item.get('vector'):
                    v_db = np.array(item['vector'])
                    sim = float(cosine_similarity(v_test.reshape(1,-1), v_db.reshape(1,-1))) * 100
                    results.append({"name": item['file_name'], "sim": sim, "url": item['img_url']})
            
            results = sorted(results, key=lambda x: x['sim'], reverse=True)[:5]
            
            st.subheader("🔥 Các mẫu tương đồng nhất:")
            cols = st.columns(len(results))
            for i, res in enumerate(results):
                with cols[i]:
                    st.image(res['url'], use_container_width=True)
                    st.write(f"**{res['name']}**")
                    st.success(f"Khớp: {res['sim']:.1f}%")
        else:
            st.warning(f"Chưa có mẫu '{target['cat']}' trong kho.")
