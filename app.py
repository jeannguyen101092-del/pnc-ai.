import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.7", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM PHÂN LOẠI CHI TIẾT (FIX THEO YÊU CẦU) =================
def classify_logic(specs, text, name):
    txt = (text + " " + name).upper()
    
    # Lấy các thông số then chốt để phân loại
    inseam = specs.get('INSEAM', 0)
    length = specs.get('LENGTH', specs.get('OUTSEAM', 0))
    sleeve = specs.get('SLEEVE', 0)
    shoulder = specs.get('SHOULDER', 0)

    # 1. NHÓM QUẦN (PANT/SHORT)
    if any(k in txt for k in ['PANT', 'CARGO', 'SHORT', 'TROUSER', 'JOGGER']):
        if 'SHORT' in txt or (0 < inseam <= 11) or (0 < length <= 22):
            return "QUẦN SHORT"
        
        # Phân biệt Lưng thun và Lưng thường
        if any(k in txt for k in ['ELASTIC', 'WAISTBAND', 'THUN', 'RIB']):
            return "QUẦN LƯNG THUN"
        return "QUẦN LƯNG THƯỜNG"

    # 2. NHÓM VÁY/ĐẦM
    if 'DRESS' in txt: return "ĐẦM / LIỀN QUẦN"
    if 'SKIRT' in txt: return "VÁY"

    # 3. NHÓM ÁO (Phân loại theo tay áo và kiểu dáng)
    if any(k in txt for k in ['VEST', 'BLAZER', 'JACKET', 'COAT']):
        return "ÁO VEST / JACKET"
    
    if any(k in txt for k in ['SHIRT', 'TEE', 'TOP', 'POLO', 'HOODIE']):
        if sleeve >= 20: 
            return "ÁO DÀI TAY"
        if 0 < sleeve <= 12: 
            return "ÁO NGẮN TAY"
        return "ÁO (KHÁC)"

    return "HÀNG KHÁC"

# ================= TRÍCH XUẤT DỮ LIỆU CÓ BỘ LỌC =================
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
                        key_found = None
                        # Mở rộng bộ từ khóa quét thông số
                        for k in ['INSEAM','WAIST','HIP','LENGTH','SLEEVE','SHOULDER','CHEST']:
                            if k in txt_r: key_found = k; break
                        if key_found:
                            vals = [parse_val(x) for x in r if x]
                            valid_vals = [v for v in vals if v >= 3]
                            if valid_vals:
                                specs[key_found] = round(float(max(valid_vals)), 2)
        
        # BỘ LỌC: Bỏ qua nếu thiếu thông số hoặc ảnh
        if len(specs) < 3: return None

        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        if not img_bytes or len(img_bytes) < 10000: return None

        return {
            "spec": specs, "img": img_bytes, 
            "cat": classify_logic(specs, text, os.path.basename(pdf_path)),
            "name": os.path.basename(pdf_path)
        }
    except: return None

# ================= CÁC HÀM XỬ LÝ (SUPABASE) =================
def compress_to_webp(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=70)
    return buf.getvalue()

def upload_to_storage(img_bytes, filename):
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename) + ".webp"
        supabase.storage.from_(BUCKET_NAME).upload(
            path=clean_name, file=img_bytes,
            file_options={"content-type": "image/webp", "upsert": "true"}
        )
        return supabase.storage.from_(BUCKET_NAME).get_public_url(clean_name)
    except: return None

# ================= GIAO DIỆN CHÍNH =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        res_count = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{res_count.count} mẫu")
    except: st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF vào kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        success, skip = 0, 0
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                img_webp = compress_to_webp(d['img'])
                img_url = upload_to_storage(img_webp, f.name)
                if img_url:
                    tf = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        vec = ai_brain(tf(Image.open(io.BytesIO(img_webp))).unsqueeze(0)).flatten().numpy().tolist()
                    supabase.table("ai_data").upsert({
                        "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                        "img_url": img_url, "category": d['cat']
                    }, on_conflict="file_name").execute()
                    success += 1
            else:
                st.warning(f"⚠️ Bỏ qua: {f.name} (Không đủ chuẩn)")
                skip += 1
        st.success(f"✅ Nạp xong: {success} | Bỏ qua: {skip}")
        st.rerun()

# ================= PHẦN TEST & EXCEL (GIỮ NGUYÊN) =================
st.title("👔 AI Fashion Pro V11.7")
test_file = st.file_uploader("Tải file PDF Test (Đối chứng)", type="pdf")
if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.write(f"### Nhận diện: **{target['cat']}**")
        # Logic hiển thị bảng so sánh và nút xuất Excel giữ nguyên như bản V11.3
