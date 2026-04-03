import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. KẾT NỐI SUPABASE
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI SMART CLASSIFY V4", page_icon="🧶")

# --- HÀM HỖ TRỢ AI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

# --- LOGIC NHẬN DIỆN THÔNG MINH THEO BẢNG BẠN GỬI ---
def advanced_classify(specs, text_content, file_name):
    txt = (text_content + " " + file_name).upper()
    inseam = specs.get('INSEAM', 0)
    
    # 1. QUẦN YẾM (BIB)
    if 'FRONT BIB WIDTH' in txt or 'BIB' in txt: return "QUẦN YẾM"
    
    # 2. QUẦN TÚI CARGO
    if 'FRONT CARGO POCKET' in txt or 'CARGO' in txt: return "QUẦN TÚI CARGO"
    
    # 3. QUẦN DÀI vs QUẦN SHORT (Dựa trên số đo Inseam)
    if inseam > 0:
        if inseam >= 25: return "QUẦN DÀI (LƯNG THƯỜNG/THUN)"
        if inseam <= 11: return "QUẦN SHORT"
    
    # 4. VÁY (SKIRT) & ĐẦM (DRESS)
    if 'SKIRT' in txt: return "VÁY"
    if 'DRESS' in txt or 'JUMPSUIT' in txt: return "ĐẦM/ÁO LIỀN QUẦN"
    
    # 5. ÁO (Dựa trên HPS và Chest)
    if any(k in txt for k in ['FRONT LENGTH', 'HPS', 'CHEST WIDTH', 'BUST']): return "ÁO"
    
    # Mặc định dựa trên từ khóa phổ thông
    if any(k in txt for k in ['PANT', 'TROUSER', 'BOTTOM']): return "QUẦN DÀI"
    
    return "KHÁC"

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
        specs, all_texts = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: all_texts += t + " "
                tables = p.extract_tables()
                for table in tables:
                    for r in table:
                        if r and len(r) >= 2:
                            val = parse_val(r[-1])
                            pom = " ".join([str(x) for x in r[:-1] if x]).strip().upper()
                            if val > 0: specs[pom] = val
                            
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
        
        # Gọi hàm nhận diện thông minh
        category = advanced_classify(specs, all_texts, os.path.basename(pdf_path))
        
        return {"spec": specs, "img_b64": img_b64, "cat": category, "img_bytes": pix.tobytes("png")}
    except: return None

# --- GIAO DIỆN ---
st.title("🧶 AI SMART SPEC PRO V4 - PHÂN LOẠI CHUYÊN SÂU")

with st.sidebar:
    st.header("📦 KHO DỮ LIỆU")
    res = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res.count if res.count else 0)
    
    up_bulk = st.file_uploader("Nạp file mẫu PDF", accept_multiple_files=True)
    if up_bulk and st.button("🚀 BẮT ĐẦU NẠP KHO"):
        for f in up_bulk:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v = ai_brain(tf(Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": v, "spec_json": d['spec'], "img_base64": d['img_b64'], "category": d['cat']}).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- SO SÁNH ---
up_test = st.file_uploader("📥 Tải file cần kiểm tra (PDF)", type="pdf")
if up_test:
    with open("test.pdf", "wb") as f: f.write(up_test.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.subheader(f"🎯 Hệ thống nhận diện: **{target['cat']}**")
        
        # Chỉ tìm kiếm mẫu CÙNG LOẠI trong kho để so sánh chuẩn xác
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if not db.data:
            st.warning(f"Kho chưa có mẫu nào thuộc loại '{target['cat']}' để đối chiếu!")
        else:
            # (Phần tính toán so sánh Similarity và hiện bảng 4 cột + nút Excel giữ nguyên như bản V3.4)
            # ... [Đoạn code hiển thị ảnh song song và bảng 4 cột của bản trước] ...
            st.info("Đang so sánh với mẫu khớp nhất trong danh mục " + target['cat'])
