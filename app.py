import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from supabase import create_client, Client
import matplotlib.pyplot as plt

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.49", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM XỬ LÝ SỐ ĐO (Hỗ trợ phân số) =================
def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        if not t_str or any(x in t_str.upper() for x in ['SIZE', 'TOL', 'DATE']): return 0
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= TRÍCH XUẤT THÔNG SỐ (PHẢI QUÉT TẤT CẢ TRANG) =================
def get_data(pdf_path):
    try:
        specs, text, base_size = {}, "", "8" # Mặc định
        
        with pdfplumber.open(pdf_path) as pdf:
            # QUAN TRỌNG: Quét tất cả các trang vì bảng Specs thường nằm cuối (Trang 11-12)
            for p in pdf.pages:
                p_text = p.extract_text() or ""
                text += p_text
                
                # Tìm Base Size linh hoạt (ví dụ: Base Size : 8-)
                m = re.search(r'Base Size\s*[:\s]\s*(\w+[\-?]?)', p_text, re.I)
                if m: base_size = m.group(1).upper()

                tables = p.extract_tables()
                for tb in tables:
                    if not tb or len(tb) < 2: continue
                    
                    # 1. Tìm hàng Header thực sự (Chứa size mục tiêu)
                    h_idx = -1
                    header = []
                    for i, row in enumerate(tb[:8]):
                        row_clean = [str(x).strip().upper() for x in row if x]
                        if any(base_size in x for x in row_clean):
                            h_idx = i
                            header = [str(x).strip().upper() for x in row]
                            break
                    
                    if h_idx == -1: continue

                    # 2. Định vị cột: Desc và Base Size
                    b_idx = -1
                    for idx, val in enumerate(header):
                        if base_size == val or (base_size in val and len(val) < 5):
                            b_idx = idx
                    
                    d_idx = 0
                    for idx, val in enumerate(header):
                        if any(k in val for k in ['DESC', 'POM', 'POINT']):
                            d_idx = idx; break

                    # 3. Lấy dữ liệu (Xử lý merged cell cho Description)
                    if b_idx != -1:
                        for r in tb[h_idx + 1:]:
                            if not r or len(r) <= max(b_idx, d_idx): continue
                            
                            # Lấy mô tả (kết hợp các cột chữ trước cột số)
                            desc = " ".join([str(r[i] or "") for i in range(b_idx) if i <= d_idx+1]).strip()
                            desc = desc.upper().replace('\n', ' ')
                            
                            if len(desc) < 5 or any(x in desc for x in ['DATE', 'PAGE', 'SIZE']): continue
                            
                            val = parse_val(r[b_idx])
                            if val > 0:
                                clean_k = re.sub(r'[^A-Z0-9\s/]', '', desc).strip()
                                specs[clean_k[:100]] = round(float(val), 3)

        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None

def classify_logic(specs, text, name):
    txt = (text + " " + name).upper()
    if 'SKORT' in txt: return "QUẦN VÁY (SKORT)"
    if 'SHORT' in txt: return "QUẦN SHORT"
    return "QUẦN DÀI" if 'PANT' in txt else "ÁO / KHÁC"

# ================= GIAO DIỆN CHÍNH =================
st.title("👔 AI Fashion Pro V11.49")

test_file = st.file_uploader("Tải PDF Spec (Ví dụ: 5651_PDM.pdf)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    data = get_data("test.pdf")
    
    if data:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(data['img'], caption="Ảnh từ PDF", use_container_width=True)
            st.success(f"Loại: {data['cat']}")
        with c2:
            st.subheader("Thông số trích xuất (Tất cả trang):")
            if data['spec']:
                df = pd.DataFrame(data['spec'].items(), columns=['Mô tả', 'Thông số'])
                st.dataframe(df, use_container_width=True, height=600)
            else:
                st.error("Vẫn không tìm thấy bảng. Vui lòng kiểm tra lại định dạng file.")
