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
def extract_pom_new_v437(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).replace({None: np.nan})
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm tọa độ cột
                    p_name_idx, val_idx = -1, -1
                    
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c is not np.nan]
                        
                        # Ưu tiên 1: Cấu trúc Reitmans (POM NAME + NEW)
                        if "POM NAME" in row_up and "NEW" in row_up:
                            p_name_idx = [str(c).upper().strip() for c in row].index("POM NAME")
                            val_idx = [str(c).upper().strip() for c in row].index("NEW")
                            break
                        
                        # Ưu tiên 2: Cấu trúc hãng khác (POM/Description + TOL + Size)
                        elif any(x in row_up for x in ["POM", "DESCRIPTION", "SPEC DESCRIPTION"]):
                            # Lấy cột tên (thường là cột đầu tiên hoặc cột chứa chữ DESCRIPTION)
                            for i, col_val in enumerate(row_up):
                                if any(x in col_val for x in ["POM", "DESCRIPTION"]): 
                                    p_name_idx = i
                                    break
                            
                            # Tìm cột giá trị (thường nằm sau cột TOL hoặc +/-)
                            for i, col_val in enumerate(row_up):
                                if "TOL" in col_val or "+/-" in col_val:
                                    val_idx = i + 1 # Cột size đầu tiên thường sau TOL
                                    break
                            
                            if p_name_idx != -1 and val_idx != -1: break

                    # Nếu tìm thấy cột, bắt đầu lấy dữ liệu
                    if p_name_idx != -1 and val_idx != -1:
                        for d_idx in range(r_idx + 1, len(df)):
                            d_row = df.iloc[d_idx]
                            name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                            
                            # Bỏ qua dòng rác
                            if len(name) < 3 or any(x in name for x in ["REF:", "RELATED:", "PAGE", "TOTAL"]): continue
                            
                            raw_val = d_row[val_idx]
                            val_num = parse_reitmans_val(raw_val)
                            
                            if val_num > 0:
                                full_specs[name] = val_num
                                
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

# --- PHẦN UI VÀ LOGIC CÒN LẠI GIỮ NGUYÊN ---
# (Copy tiếp các phần SIDEBAR và MAIN của bạn vào đây)
