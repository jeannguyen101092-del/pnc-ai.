import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (HÃY ĐIỀN THÔNG TIN CỦA BẠN) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.14", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= TRÍCH XUẤT TOÀN BỘ THÔNG SỐ (FIX CHÍ MẠNG) =================
def parse_val(t):
    try:
        # Xử lý phân số như 30 1/2 hoặc 1/4
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
                    # --- BỘ LỌC BẢNG POM (CHỐNG QUÉT NHẦM BOM) ---
                    # Chuyển bảng thành chữ để kiểm tra từ khóa
                    table_content = str(tb).upper()
                    
                    # Nếu thấy các chữ liên quan đến Phụ liệu (BOM) thì BỎ QUA NGAY
                    if any(x in table_content for x in ['FABRIC', 'MATERIAL', 'THREAD', 'BUTTON', 'ZIPPER', 'BOM']):
                        continue
                        
                    # CHỈ LẤY bảng nào có các từ khóa đo đạc (POM)
                    if not any(x in table_content for x in ['WAIST', 'HIP', 'INSEAM', 'THIGH', 'LENGTH', 'POM', 'SPEC']):
                        continue

                    for r in tb:
                        if not r or len(r) < 2: continue
                        clean_row = [str(x).strip() for x in r if x]
                        label = str(clean_row[0]).upper()
                        
                        # Lấy các số đo thực tế (Inches thường từ 3.0 đến 60.0)
                        # Bỏ qua các số quá lớn (>100) vì đó là mã phụ liệu
                        vals = [parse_val(x) for x in clean_row[1:] if 2.5 <= parse_val(x) <= 120.0]
                        
                        if vals and len(label) > 2:
                            specs[label[:50]] = round(float(vals[0]), 2)
                            
        doc = fitz.open(pdf_path)
        # TĂNG CHẤT LƯỢNG ẢNH: Để bạn nhìn bản vẽ POM rõ hơn
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.5, 2.5)).tobytes("png")
        doc.close()
        
        # Nhận diện loại dựa trên POM thực
        cat = "ÁO"
        if any(x in str(specs.keys()) for x in ['INSEAM', 'THIGH', 'HIP']):
            cat = "QUẦN DÀI LƯNG THƯỜNG"
            if 'ELASTIC' in text.upper(): cat = "QUẦN DÀI LƯNG THUN"
            if 0 < specs.get('INSEAM', 0) < 15: cat = "QUẦN SHORT"

        return {"spec": specs, "img": img, "cat": cat, "name": os.path.basename(pdf_path)}
    except: return None

# ================= SIDEBAR & QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []
        st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF mới", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d and len(d['spec']) > 3: # Chỉ nạp nếu lấy được trên 3 thông số
                img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                buf = io.BytesIO()
                img_p.save(buf, format="WEBP", quality=75)
                fname = re.sub(r'[^a-zA-Z0-9]', '_', f.name) + ".webp"
                supabase.storage.from_(BUCKET_NAME).upload(path=fname, file=buf.getvalue(), file_options={"upsert":"true"})
                url = supabase.storage.from_(BUCKET_NAME).get_public_url(fname)
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": vec, "spec_json": d['spec'], "img_url": url, "category": d['cat']}, on_conflict="file_name").execute()
        st.success("🏁 Nạp xong!")
        st.rerun()

# ================= CHÍNH: SO SÁNH TOÀN DIỆN =================
st.title("👔 AI Fashion Pro V11.14")
test_file = st.file_uploader("Tải file PDF Test (Đối chứng)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        
        list_names = [item['file_name'] for item in all_samples]
        selected_code = st.selectbox("🎯 Chọn mã hàng cụ thể (hoặc AI tự tìm):", ["-- Tự động tìm mẫu tương đồng --"] + list_names)
        
        matches = []
        if selected_code != "-- Tự động tìm mẫu tương đồng --":
            for item in all_samples:
                if item['file_name'] == selected_code:
                    matches = [{"name": item['file_name'], "sim": 100.0, "url": item['img_url'], "spec": item['spec_json']}]
                    break
        else:
            if all_samples:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
                for item in all_samples:
                    if item.get('vector'):
                        v_db = np.array(item['vector']).reshape(1, -1)
                        sim_val = float(cosine_similarity(v_test.reshape(1, -1), v_db)[0][0]) * 100
                        matches.append({"name": item['file_name'], "sim": sim_val, "url": item['img_url'], "spec": item['spec_json']})
                matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]

        if matches:
            for m in matches:
                with st.expander(f"📌 ĐỐI CHIẾU TOÀN DIỆN: {m['name']} (Giống {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.5])
                    with c1: st.image(target['img'], caption="Bản vẽ PDF Test")
                    with c2: st.image(m['url'], caption="Bản vẽ trong Kho")
                    with c3:
                        # Gộp tất cả thông số của cả 2 bên để so sánh
                        all_keys = sorted(list(set(target['spec'].keys()).union(set(m['spec'].keys()))))
                        comp_list = []
                        for k in all_keys:
                            v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                            diff = round(v_t - v_d, 2)
                            comp_list.append({"Điểm đo": k, "Test": v_t, "Kho": v_d, "Lệch": diff})
                        
                        df_res = pd.DataFrame(comp_list)
                        st.table(df_res.style.map(lambda x: 'color: red' if abs(x) > 0.25 else 'color: green', subset=['Lệch']))
        else: st.warning("Chưa có dữ liệu so sánh.")
