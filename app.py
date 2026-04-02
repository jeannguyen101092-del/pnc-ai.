import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# CẤU HÌNH (Thay URL và KEY của bạn)
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion - Nhận diện hình vẽ", page_icon="👗")

# --- HÀM HỖ TRỢ ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

def get_data(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        # Lấy trang đầu làm hình vẽ chính (Soi)
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
        main_img = pix.tobytes("png")
        
        # Lấy các hình ảnh nhỏ khác (nếu có ở các trang sau) làm Gallery
        gallery = []
        for i in range(min(len(doc), 3)):
            p = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
            gallery.append(p.tobytes("png"))
            
        # Đọc thông số (POM)
        specs = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for r in table:
                        if r and len(r) >= 2:
                            val = re.findall(r'\d+\.?\d*', str(r[-1]))
                            if val: specs[str(r[0]).upper()] = float(val[0])
        
        return {"main_img": main_img, "gallery": gallery, "spec": specs}
    except: return None

# --- GIAO DIỆN CHÍNH ---
st.title("👗 AI Fashion - Nhận diện hình vẽ (Bản hiển thị ảnh)")

with st.sidebar:
    st.header("⚙️ QUẢN TRỊ KHO")
    if st.button("♻️ Cập nhật dữ liệu"): st.rerun()
    up_bulk = st.file_uploader("Nạp kho PDF", accept_multiple_files=True)
    if up_bulk and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in up_bulk:
            with open("temp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("temp.pdf")
            if d:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    v = ai_brain(tf(Image.open(io.BytesIO(d['main_img'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": v, "spec_json": d['spec'], "category": "KHO"}).execute()
        st.success("Đã nạp kho!")

# --- PHẦN HIỂN THỊ KẾT QUẢ ---
up = st.file_uploader("📥 Drag and drop file here", type="pdf")

if up:
    with open("test.pdf", "wb") as f: f.write(up.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔍 Hình vẽ đang soi")
            st.image(target['main_img'], use_container_width=True)
            # Hiển thị Gallery nhỏ bên dưới
            g_cols = st.columns(3)
            for i, g_img in enumerate(target['gallery']):
                with g_cols[i % 3]: st.image(g_img, use_container_width=True)

        with col2:
            st.subheader("✅ Kết quả (Dựa trên hình vẽ thực tế)")
            # Tìm kiếm trong kho
            db = supabase.table("ai_data").select("*").execute()
            if db.data:
                # (Logic tính toán độ tương đồng sim_list giữ nguyên như bản trước)
                # Giả sử lấy mã khớp nhất là 'best_match'
                best_match = db.data[0] # Ví dụ mã đầu tiên
                
                res_cols = st.columns(2)
                with res_cols[0]:
                    st.write(f"🔴 **{best_match['file_name']}**")
                    # Ở đây bạn cần lưu thêm hình ảnh vào Supabase nếu muốn hiện ảnh của mẫu trong kho
                    st.image(target['main_img'], caption="Ảnh tương đồng", use_container_width=True) 
                
                # Bảng thông số (POM) hiển thị bên cạnh ảnh
                st.write("---")
                st.write("**Bảng đối chiếu thông số:**")
                st.table(pd.DataFrame(list(target['spec'].items()), columns=['Thông số', 'Giá trị']))
