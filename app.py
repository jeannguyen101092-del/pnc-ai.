import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc
from torchvision import models, transforms
from supabase import create_client, Client
import matplotlib.pyplot as plt

# ================= CONFIG (Thay URL và KEY của bạn) =================
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

# ================= HÀM CHỤP ẢNH EXCEL ĐỊNH MỨC =================
def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).dropna(how='all', axis=0).fillna("")
        df_display = df.head(80)
        fig, ax = plt.subplots(figsize=(24, len(df_display) * 0.6 + 2)) 
        ax.axis('off')
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(16) 
        table.scale(1.2, 3.2) 
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white', size=18)
                cell.set_facecolor('#000000')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3, dpi=300)
        plt.close(fig)
        return buf.getvalue()
    except: return None

# ================= HÀM XỬ LÝ SỐ ĐO (Hỗ trợ phân số) =================
def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        if not t_str or any(x in t_str.upper() for x in ['SIZE', 'TOL', 'DATE']): return 0
        # Tìm các định dạng số: 15 1/2, 15.5, 15
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= TRÍCH XUẤT THÔNG SỐ (BẢN TỐI ƯU) =================
def get_data(pdf_path):
    try:
        specs, text, base_size = {}, "", "8" # Mặc định là size 8
        
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text() or ""
                text += page_text
                
                # Tìm Base Size trong text (VD: Base Size: 10 hoặc Base Size 8)
                m = re.search(r'Base Size\s*[:\s]\s*(\w+)', page_text, re.IGNORECASE)
                if m: base_size = m.group(1).upper()

                tables = p.extract_tables()
                for tb in tables:
                    if not tb or len(tb) < 2: continue
                    
                    # 1. Tìm hàng Header thực sự (Chứa size)
                    header_row_idx = -1
                    header = []
                    for i, row in enumerate(tb[:5]): # Quét 5 hàng đầu bảng
                        row_clean = [str(x).strip().upper() for x in row if x]
                        if any(base_size == x or f" {base_size} " in f" {x} " for x in row_clean):
                            header_row_idx = i
                            header = [str(x).strip().upper() for x in row]
                            break
                    
                    if header_row_idx == -1: continue

                    # 2. Định vị cột: Mô tả (Desc) và Thông số (Base Size)
                    try:
                        # Cột Size (Lấy cột cuối cùng khớp để tránh cột dung sai)
                        base_idx = -1
                        for idx, h_val in enumerate(header):
                            if base_size == h_val or f" {base_size} " in f" {h_val} ":
                                base_idx = idx
                        
                        # Cột Mô tả (Thường là 'Description', 'Point of Measure', hoặc cột đầu tiên có chữ)
                        desc_idx = 0
                        for idx, h_val in enumerate(header):
                            if any(k in h_val for k in ['DESC', 'POINT', 'MEASURE']):
                                desc_idx = idx
                                break
                    except: continue

                    if base_idx != -1:
                        for r in tb[header_row_idx + 1:]:
                            if not r or len(r) <= max(base_idx, desc_idx): continue
                            
                            # Lấy tên thông số (Description)
                            desc = str(r[desc_idx] or "").strip().upper().replace('\n', ' ')
                            if not desc and desc_idx + 1 < len(r): 
                                desc = str(r[desc_idx+1] or "").strip().upper()
                            
                            # Lọc rác: Bỏ các dòng header lặp lại hoặc quá ngắn
                            if len(desc) < 4 or any(x in desc for x in ['DATE', 'PAGE', 'TOLERANCE']): continue
                            
                            val = parse_val(r[base_idx])
                            if val > 0.1: # Chỉ lấy thông số có giá trị
                                # Làm sạch key: Giữ lại chữ và số
                                clean_key = re.sub(r'[^A-Z0-9\s/]', '', desc).strip()
                                specs[clean_key[:100]] = round(float(val), 3)

        # Chụp ảnh trang đầu PDF
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except Exception as e:
        print(f"Error parsing: {e}")
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

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF & Excel (Cùng mã số)", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        groups = {}
        for f in files:
            m = re.search(r'^\d+', f.name)
            if m:
                ma = m.group()
                ext = os.path.splitext(f.name)[1].lower()
                if ma not in groups: groups[ma] = {}
                groups[ma][ext] = f
        
        for ma, parts in groups.items():
            f_p = parts.get('.pdf')
            f_e = parts.get('.xlsx') or parts.get('.xls')
            if f_p and f_e:
                with st.spinner(f"Đang xử lý mã: {ma}"):
                    with open("tmp.pdf", "wb") as t: t.write(f_p.getbuffer())
                    d = get_data("tmp.pdf")
                    exl = excel_to_img_bytes(f_e)
                    
                    if d and exl:
                        img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                        buf_img = io.BytesIO(); img_p.save(buf_img, format="WEBP")
                        
                        try:
                            # Upload ảnh PDF và ảnh Excel lên Supabase Storage
                            supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_t.webp", buf_img.getvalue(), {"content-type": "image/webp", "x-upsert": "true"})
                            supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_e.webp", exl, {"content-type": "image/webp", "x-upsert": "true"})
                            
                            url_t = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_t.webp")
                            url_e = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_e.webp")
                            
                            # Vector hóa ảnh để tìm kiếm thông minh
                            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                            with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                            
                            supabase.table("ai_data").upsert({
                                "file_name": ma, "vector": vec, "spec_json": d['spec'], 
                                "img_url": url_t, "excel_img_url": url_e, "category": d['cat']
                            }).execute()
                        except Exception as e: st.error(f"Lỗi nạp mã {ma}: {e}")
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.success("Đã nạp xong!")
        st.rerun()

# ================= MAIN UI: SO SÁNH =================
st.title("👔 AI Fashion Pro V11.49")
test_file = st.file_uploader("Tải PDF Spec cần kiểm tra", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    data_test = get_data("test.pdf")
    
    if data_test:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(data_test['img'], caption="Ảnh từ PDF Test", use_container_width=True)
            st.write(f"**Loại sản phẩm:** {data_test['cat']}")
        
        with col2:
            st.write("### Thông số trích xuất được:")
            st.dataframe(pd.DataFrame(data_test['spec'].items(), columns=['Mô tả', 'Số đo (Base Size)']), height=400)
    else:
        st.error("Không thể trích xuất dữ liệu từ PDF này.")
