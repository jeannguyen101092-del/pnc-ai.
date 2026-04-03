import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc
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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.21", page_icon="👔")

# ================= AI ENGINE (MOBILENET V2 - SIÊU NHẸ) =================
@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM XỬ LÝ (CHỈ TRÍCH XUẤT - KHÔNG LƯU PDF) =================
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

def classify_logic(specs, text, name):
    txt = (text + " " + name).upper()
    inseam = specs.get('INSEAM', 0)
    length = specs.get('LENGTH', specs.get('OUTSEAM', 0))
    if any(k in txt for k in ['PANT', 'CARGO', 'TROUSER']) or length >= 25 or inseam >= 15:
        if any(k in txt for k in ['ELASTIC WAIST', 'RIB WAIST', 'FULL ELASTIC']):
            return "QUẦN DÀI LƯNG THUN"
        return "QUẦN DÀI LƯNG THƯỜNG"
    if 0 < length <= 23 or 0 < inseam <= 13 or 'SHORT' in txt: return "QUẦN SHORT"
    return "ÁO"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        # 1. Quét thông số xong là đóng file ngay
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    content = str(tb).upper()
                    if any(x in content for x in ['FABRIC', 'MATERIAL', 'BOM']): continue
                    for r in tb:
                        if not r or len(r) < 2: continue
                        raw_label = " ".join([str(x) for x in r[:2] if x]).strip().upper()
                        clean_label = re.sub(r'^[A-Z]\d{1,4}[A-Z]?(\.\d+)?\s*', '', raw_label)
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(clean_label) > 3:
                            specs[clean_label[:60]] = round(float(np.median(vals)), 2)
        
        # 2. Chụp ảnh trang đầu rồi đóng file PDF ngay lập tức
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= SIDEBAR: NẠP KHO (CHỈ LƯU ẢNH) =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Upload PDF (AI sẽ chỉ lấy ảnh & thông số)", accept_multiple_files=True)
    
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for idx, f in enumerate(files):
            p_bar.progress((idx + 1) / len(files))
            # Lưu tạm để xử lý
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            
            if d:
                # Nén ảnh thành WebP cực nhẹ
                img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                buf = io.BytesIO()
                img_p.save(buf, format="WEBP", quality=60)
                
                # CHỈ UPLOAD ẢNH LÊN STORAGE
                fname = re.sub(r'[^a-zA-Z0-9]', '_', f.name) + ".webp"
                supabase.storage.from_(BUCKET_NAME).upload(path=fname, file=buf.getvalue(), file_options={"upsert":"true"})
                url = supabase.storage.from_(BUCKET_NAME).get_public_url(fname)
                
                # Trích xuất Vector AI
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                # LƯU DỮ LIỆU CHỮ VÀ LINK ẢNH VÀO DATABASE (KHÔNG LƯU PDF)
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['spec'], 
                    "img_url": url, "category": d['cat']
                }, on_conflict="file_name").execute()
            
            # Xóa file tạm ngay sau khi xử lý xong 1 file
            if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
            gc.collect() 
            
        st.success("🏁 Nạp kho thành công (Đã tối ưu dung lượng)!"); st.rerun()

# ================= CHÍNH: SO SÁNH =================
st.title("👔 AI Fashion Pro V11.21")
test_file = st.file_uploader("Tải file PDF Test để đối chiếu", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        list_names = [item['file_name'] for item in all_samples]
        selected = st.selectbox("🎯 Chọn mã hàng cụ thể (hoặc để AI tự tìm):", ["-- Tự động tìm mẫu tương đồng --"] + list_names)
        
        matches = []
        if selected == "-- Tự động tìm mẫu tương đồng --":
            if all_samples:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
                for i in all_samples:
                    if i.get('vector'):
                        v_db = np.array(i['vector']).reshape(1, -1)
                        sim_val = float(cosine_similarity(v_test.reshape(1, -1), v_db)) * 100
                        matches.append({"name": i['file_name'], "sim": sim_val, "url": i['img_url'], "spec": i['spec_json']})
                matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
        else:
            for i in all_samples:
                if i['file_name'] == selected:
                    matches = [{"name": i['file_name'], "sim": 100.0, "url": i['img_url'], "spec": i['spec_json']}]
                    break

        if matches:
            for m in matches:
                with st.expander(f"📌 ĐỐI CHIẾU: {m['name']} (Giống {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Bản vẽ Test")
                    with c2: st.image(m['url'], caption="Ảnh mẫu trong kho")
                    with c3:
                        all_k = sorted(list(set(target['spec'].keys()).union(set(m['spec'].keys()))))
                        comp = [{"Thông số": k, "Test": target['spec'].get(k, 0), "Kho": m['spec'].get(k, 0), "Lệch": round(target['spec'].get(k, 0) - m['spec'].get(k, 0), 2)} for k in all_k]
                        st.table(pd.DataFrame(comp).style.format(subset=['Test', 'Kho', 'Lệch'], precision=2).map(lambda x: 'color: red' if abs(x) > 0.25 else 'color: green', subset=['Lệch']))
    
    if os.path.exists("test.pdf"): os.remove("test.pdf")
    gc.collect()
