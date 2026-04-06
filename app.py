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

# ================= HÀM XỬ LÝ SỐ ĐO (Hỗ trợ phân số ngành may) =================
def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        if not t_str or any(x in t_str.upper() for x in ['SIZE', 'TOL', 'DATE']): return 0
        # Tìm định dạng: 15 1/2, 15.5, 15
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= HÀM TRÍCH XUẤT THÔNG SỐ SIÊU CẤP =================
def get_data(pdf_path):
    try:
        specs, text, base_size = {}, "", "8" # Mặc định size 8
        
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text() or ""
                text += page_text
                
                # Tìm Base Size linh hoạt (Base Size: 8, Sample Size 10, v.v.)
                m = re.search(r'(?:Base Size|Sample Size|Size)\s*[:\s]\s*(\w+)', page_text, re.I)
                if m: base_size = m.group(1).upper()

                tables = p.extract_tables()
                for tb in tables:
                    if not tb or len(tb) < 2: continue
                    
                    # 1. Tìm hàng Header thực sự (Quét 8 hàng đầu để tránh Logo/Header rác)
                    h_idx = -1
                    header = []
                    for i, row in enumerate(tb[:8]):
                        row_clean = [str(x).strip().upper() for x in row if x]
                        # Tìm hàng chứa đúng tên Base Size
                        if any(base_size == x or f" {base_size} " in f" {x} " for x in row_clean):
                            h_idx = i
                            header = [str(x).strip().upper() for x in row]
                            break
                    
                    if h_idx == -1: continue # Không thấy size ở bảng này, bỏ qua

                    # 2. Định vị cột: Mô tả (Desc) và Thông số (Base Size)
                    # Lấy cột cuối cùng khớp Size (tránh cột Tolerance/Dung sai phía trước)
                    b_idx = -1
                    for idx, val in enumerate(header):
                        if base_size == val or f" {base_size} " in f" {val} ":
                            b_idx = idx
                    
                    # Cột Mô tả (Ưu tiên cột có chữ 'Description', 'Point', 'Measure')
                    d_idx = 0
                    for idx, val in enumerate(header):
                        if any(k in val for k in ['DESC', 'POINT', 'MEASURE', 'POM']):
                            d_idx = idx; break

                    # 3. Quét dữ liệu từ hàng sau Header
                    if b_idx != -1:
                        for r in tb[h_idx + 1:]:
                            if not r or len(r) <= max(b_idx, d_idx): continue
                            
                            # Lấy tên thông số (Nếu cột d_idx trống, thử lấy cột kế bên - xử lý merged cell)
                            desc = str(r[d_idx] or "").strip().upper().replace('\n', ' ')
                            if len(desc) < 3 and d_idx + 1 < len(r):
                                desc = str(r[d_idx+1] or "").strip().upper()
                            
                            val = parse_val(r[b_idx])
                            # Lọc rác: Bỏ dòng tiêu đề lặp lại hoặc dòng ghi chú
                            if val > 0.1 and len(desc) > 3 and not any(x in desc for x in ['DATE', 'PAGE', 'SIZE']):
                                clean_key = re.sub(r'[^A-Z0-9\s/]', '', desc).strip()
                                specs[clean_key[:100]] = round(float(val), 3)

        # Chụp ảnh Preview trang 1
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        
        return {"spec": specs, "img": img_bytes, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except Exception as e:
        st.error(f"Lỗi phân tích PDF: {e}")
        return None

def classify_logic(specs, text, name):
    txt = (text + " " + name).upper()
    length = 0
    if specs:
        lv = [v for k,v in specs.items() if any(x in k for x in ['LENGTH', 'OUTSEAM', 'INSEAM'])]
        if lv: length = max(lv)
    if 'SHORT' in txt or (0 < length < 24): return "QUẦN SHORT"
    if any(k in txt for k in ['PANT', 'TROUSER', 'JEAN']) or length >= 24: return "QUẦN DÀI"
    return "ÁO / KHÁC"

def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).dropna(how='all', axis=0).fillna("")
        fig, ax = plt.subplots(figsize=(20, len(df.head(50)) * 0.6 + 2))
        ax.axis('off')
        table = ax.table(cellText=df.head(50).values, colLabels=df.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)
        return buf.getvalue()
    except: return None

# ================= GIAO DIỆN CHÍNH =================
st.title("👔 AI Fashion Pro V11.49")

# --- Sidebar: Nạp dữ liệu ---
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    files = st.file_uploader("Nạp PDF & Excel", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            if f.name.lower().endswith('.pdf'):
                with open("temp.pdf", "wb") as tmp: tmp.write(f.getbuffer())
                d = get_data("temp.pdf")
                if d and d['spec']:
                    # Logic Upsert Supabase ở đây (giữ nguyên như code cũ của bạn)
                    st.success(f"Đã trích xuất {len(d['spec'])} thông số từ {f.name}")
                else:
                    st.warning(f"Không tìm thấy bảng thông số trong {f.name}")

# --- Main UI: Kiểm tra ---
test_file = st.file_uploader("Tải PDF Spec cần kiểm tra", type="pdf", key="tester")
if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    data_test = get_data("test.pdf")
    
    if data_test:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(data_test['img'], caption="Ảnh thiết kế", use_container_width=True)
            st.info(f"Phân loại: {data_test['cat']}")
        with c2:
            st.subheader("Thông số trích xuất được:")
            if data_test['spec']:
                df_spec = pd.DataFrame(data_test['spec'].items(), columns=['Mô tả', 'Số đo (Base)'])
                st.dataframe(df_spec, use_container_width=True, height=500)
            else:
                st.error("⚠️ Không lấy được thông số. Hãy kiểm tra xem file PDF có bảng không.")
