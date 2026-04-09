# ==========================================================
# TOOL QUẢN LÝ MÃ HÀNG NỘI BỘ - AI SEARCH V1
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- KẾT NỐI DATABASE ---
# Thay thế URL và KEY của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="Hệ Thống Lưu Trữ Mã Hàng", page_icon="📦")

# --- LOAD AI MODEL (So sánh ảnh) ---
@st.cache_resource
def load_ai():
    # Sử dụng ResNet18 để tạo đặc trưng ảnh (Vector)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1])) # Bỏ lớp phân loại cuối
    return model.eval()

ai_brain = load_ai()

def get_image_embedding(img_bytes):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    with torch.no_grad():
        vector = ai_brain(tf(img).unsqueeze(0)).flatten().numpy()
    return vector

# --- HÀM XỬ LÝ FILE PDF ---
def process_pdf(pdf_file):
    try:
        # 1. Tách trang đầu tiên làm hình ảnh (Trang bìa/Hình mẫu)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        first_page = doc.load_page(0)
        pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Zoom 2x cho nét
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        # 2. Trích xuất thông số (Quét tất cả các trang có bảng)
        all_specs = {}
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        # Logic đơn giản: Cột 0 là tên thông số, Cột 1 là giá trị
                        if row and len(row) >= 2 and row[0] and row[1]:
                            key = str(row[0]).strip().upper()
                            val = str(row[1]).strip()
                            # Chỉ lấy các dòng có số
                            if any(char.isdigit() for char in val):
                                all_specs[key] = val
        
        doc.close()
        return {"img_bytes": img_bytes, "img_b64": img_b64, "specs": all_specs}
    except Exception as e:
        st.error(f"Lỗi xử lý PDF: {e}")
        return None

# ================= GIAO DIỆN CHÍNH =================
menu = ["Tìm kiếm mã hàng", "Nạp kho dữ liệu"]
choice = st.sidebar.selectbox("Chức năng", menu)

if choice == "Nạp kho dữ liệu":
    st.header("📤 Tải lên mã hàng mới")
    uploaded_files = st.file_uploader("Chọn các file PDF mã hàng", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Lưu vào kho"):
        for f in uploaded_files:
            data = process_pdf(f)
            if data:
                vector = get_image_embedding(data['img_bytes']).tolist()
                # Lưu vào Supabase
                res = supabase.table("product_warehouse").upsert({
                    "file_name": f.name,
                    "img_base64": data['img_b64'],
                    "specs": data['specs'],
                    "vector": vector
                }).execute()
                st.success(f"Đã lưu: {f.name}")

elif choice == "Tìm kiếm mã hàng":
    st.header("🔍 Tìm kiếm mã hàng tương đồng")
    test_file = st.file_uploader("Đẩy file PDF mới để đối chiếu", type="pdf")
    
    if test_file:
        current_data = process_pdf(test_file)
        if current_data:
            st.subheader("Dữ liệu file vừa tải lên")
            col1, col2 = st.columns([1, 2])
            col1.image(current_data['img_bytes'], caption="Ảnh mẫu trang đầu")
            col2.write("Thông số trích xuất được:")
            col2.json(current_data['specs'])

            # Lấy vector ảnh để so sánh
            v_test = get_image_embedding(current_data['img_bytes'])

            # Lấy toàn bộ kho dữ liệu từ DB (Có thể tối ưu bằng cách search vector trên DB nếu dữ liệu lớn)
            db = supabase.table("product_warehouse").select("*").execute()
            
            if db.data:
                matches = []
                for item in db.data:
                    v_db = np.array(item['vector'])
                    # Tính độ giống nhau của ảnh (0 đến 100%)
                    img_sim = float(cosine_similarity([v_test], [v_db])[0][0]) * 100
                    
                    matches.append({
                        "name": item['file_name'],
                        "sim": img_sim,
                        "img": item['img_base64'],
                        "specs": item['specs']
                    })
                
                # Sắp xếp kết quả giống nhất lên đầu
                matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:5]

                st.divider()
                st.subheader("Kết quả tìm kiếm giống nhất:")
                
                for m in matches:
                    with st.container():
                        c1, c2, c3 = st.columns([1, 1, 2])
                        c1.image(base64.b64decode(m['img']), caption=f"Mã: {m['name']}")
                        c2.metric("Độ giống nhau", f"{m['sim']:.2f}%")
                        c3.write("Thông số kỹ thuật trong kho:")
                        c3.json(m['specs'])
                        st.divider()
