import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, datetime, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ================= CONFIG (Thay URL và KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"           
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Database!")

st.set_page_config(layout="wide", page_title="AI FASHION PRO V16.2", page_icon="🛡️")

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 1500

@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

# ================= HỆ THỐNG PHÂN TÍCH =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): 
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist() # Chuyển sang list để Supabase nhận
    except: return None

def analyze_garment_logic(text):
    t = str(text).upper()
    details = []
    if 'CARGO' in t: details.append("📦 Túi Hộp (Cargo Pocket)")
    if 'SLANT' in t: details.append("📐 Túi Xéo (Slant Pocket)")
    if 'SCOOP' in t or 'HAM ECH' in t: details.append("🐸 Túi Hàm Ếch (Scoop)")
    if 'PATCH' in t: details.append("🎨 Túi Đắp (Patch Pocket)")
    if 'ELASTIC' in t: details.append("🧶 Lưng Thun (Elastic Waist)")
    if 'SKORT' in t: details.append("👗 Quần Váy (Skort)")
    if 'LONG SLEEVE' in t: details.append("🧥 Áo Dài Tay")
    if 'SHORT SLEEVE' in t: details.append("👕 Áo Ngắn Tay")
    return details

def excel_to_img_matrix(file_obj):
    try:
        ext = file_obj.name.split('.')[-1].lower()
        engine = 'xlrd' if ext == 'xls' else 'openpyxl'
        df = pd.read_excel(file_obj, engine=engine).dropna(how='all', axis=0).fillna("")
        fig, ax = plt.subplots(figsize=(24, len(df.head(80)) * 0.7 + 2))
        ax.axis('off')
        ax.table(cellText=df.head(80).values, colLabels=df.columns, loc='center', cellLoc='left').scale(1.2, 3)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150); plt.close(fig)
        return buf.getvalue()
    except: return None

def extract_pdf_ultimate(pdf_path):
    specs, text = {}, ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""; text += txt
                for tb in page.extract_tables():
                    for row in tb:
                        if len(row) >= 2: specs[str(row[0])[:50]] = str(row[1])
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU V16.2")
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res else []
    except: samples = []
    
    st.metric("TỔNG MẪU TRONG KHO", len(samples))
    list_ma = [s['file_name'] for s in samples]
    sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU", ["-- Click chọn --"] + list_ma)
    if sel != "-- Click chọn --": 
        st.session_state.target = next(s for s in samples if s['file_name'] == sel)

    st.divider()
    up_files = st.file_uploader("Nạp PDF & Excel mới", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'], key=st.session_state.up_key)
    
    if up_files and st.button("🚀 NẠP & PHÂN TÍCH CHI TIẾT AI"):
        pdfs = [f for f in up_files if f.name.lower().endswith('.pdf')]
        exls = [f for f in up_files if f.name.lower().endswith(('.xls', '.xlsx'))]
        
        for f_p in pdfs:
            # FIX LOGIC: Nếu chỉ up 1 cặp file, tự động lấy tên file PDF làm mã số
            ma = f_p.name.split('.')[0] 
            f_e = exls[0] if exls else None # Lấy file excel đầu tiên nếu có
            
            with st.spinner(f"Đang nạp mã {ma}..."):
                with open("tmp.pdf", "wb") as t: t.write(f_p.getbuffer())
                d, ex_img = extract_pdf_ultimate("tmp.pdf"), excel_to_img_matrix(f_e) if f_e else None
                
                if d:
                    vec = get_vector(d['img'])
                    det = analyze_garment_logic(d)
                    try:
                        # Upload Storage
                        supabase.storage.from_(BUCKET).upload(f"{ma}_t.png", d['img'], {"x-upsert": "true", "content-type": "image/png"})
                        u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_t.png")
                        u_e = ""
                        if ex_img:
                            supabase.storage.from_(BUCKET).upload(f"{ma}_e.png", ex_img, {"x-upsert": "true", "content-type": "image/png"})
                            u_e = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_e.png")
                        
                        # Lưu DB
                        supabase.table("ai_data").upsert({
                            "file_name": ma, "vector": vec, "spec_json": d['spec'], 
                            "img_url": u_t, "excel_img_url": u_e, "details": det
                        }).execute()
                        st.toast(f"✅ Đã lưu mã {ma}")
                    except Exception as e: st.error(f"Lỗi DB: {e}")
        st.session_state.up_key += 1; st.rerun()

# ================= MAIN UI =================
st.title("🛡️ AI FASHION PRO - SO SÁNH SIÊU CẤP")
test_pdf = st.file_uploader("1. Tải PDF Test (File cần kiểm tra)", type="pdf")
target = st.session_state.target

if test_pdf:
    with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
    data_test = extract_pdf_ultimate("test.pdf")
    
    if data_test:
        col1, col2 = st.columns(2)
        with col1: st.image(data_test['img'], caption="FILE TEST")
        
        if target:
            with col2: st.image(target['img_url'], caption=f"KHO GỐC: {target['file_name']}")
            
            # --- PHẦN SO SÁNH & XUẤT FILE ---
            st.divider()
            st.subheader("📏 BẢNG ĐỐI CHIẾU")
            res_list = []
            for k, v in data_test['spec'].items():
                v_goc = target['spec_json'].get(k, "---")
                res_list.append({"Hạng mục": k, "Test": v, "Gốc": v_goc, "Kết quả": "✅" if v == v_goc else "❌"})
            
            df_res = pd.DataFrame(res_list)
            st.table(df_res)
            
            # NÚT XUẤT FILE
            st.download_button(
                label="📥 XUẤT FILE BÁO CÁO (CSV)",
                data=df_res.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"Report_{target['file_name']}.csv",
                mime='text/csv'
            )
