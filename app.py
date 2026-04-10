import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Giữ nguyên của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.8", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- XỬ LÝ SỐ (Cải tiến để đọc được cả phân số và số thập phân) ---
def parse_reitmans_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null']: return 0
        # Tìm số, phân số (ví dụ: 1 1/2 hoặc 1/2 hoặc 15.5)
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT THÔNG SỐ ĐA DẠNG (NÂNG CẤP) ---
# --- HÀM TRÍCH XUẤT CẢI TIẾN ---
def extract_pom_new_v437(pdf_file):
    full_specs, img_bytes, brand = {}, None, "GENERIC"
    try:
        pdf_content = pdf_file.read()
        # 1. Lấy ảnh trang đầu làm thumbnail (giữ nguyên logic cũ)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        # 2. Xác định Brand/Loại hàng sơ bộ qua text
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        elif "WALMART" in all_text: brand = "WALMART"
        elif "TARGET" in all_text: brand = "TARGET"
        doc.close()

        # 3. Quét bảng từ PDF
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                # Kiểm tra xem trang này có phải trang POM không
                page_text = page.extract_text().upper() if page.extract_text() else ""
                if not any(k in page_text for k in ["POM", "SPECIFICATION", "MEASUREMENT", "TOLERANCE"]):
                    continue # Bỏ qua trang không liên quan

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm tọa độ cột: Cột mô tả (Desc) và Cột thông số (Value)
                    desc_idx, val_idx = -1, -1
                    
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # Tìm cột Description (hoặc POM Name)
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["DESCRIPTION", "POM NAME", "ITEM", "POINT OF MEASURE"]):
                                desc_idx = i
                            if any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", "TOTAL"]):
                                val_idx = i
                        
                        # Nếu tìm thấy Header thì bắt đầu lấy dữ liệu từ dòng tiếp theo
                        if desc_idx != -1 and val_idx != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[desc_idx]).replace('\n',' ').strip().upper()
                                
                                # Lọc bỏ rác
                                if len(name) < 3 or any(x in name for x in ["REF:", "DATE", "PAGE", "TOTAL"]): 
                                    continue
                                    
                                raw_val = d_row[val_idx]
                                val_num = parse_reitmans_val(raw_val) # Dùng hàm parse cũ vẫn rất tốt
                                
                                if val_num > 0:
                                    full_specs[name] = val_num
                            break # Đã xong bảng này, chuyển bảng/trang khác
                            
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except Exception as e:
        print(f"Lỗi: {e}")
        return None
