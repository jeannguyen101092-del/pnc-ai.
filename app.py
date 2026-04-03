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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.18", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM QUÉT DỮ LIỆU TOÀN BỘ POM =================
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

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    content_str = str(tb).upper()
                    # Bỏ qua bảng phụ liệu BOM
                    if any(x in content_str for x in ['FABRIC', 'MATERIAL', 'THREAD', 'BOM']): continue
                    
                    for r in tb:
                        if not r or len(r) < 2: continue
                        # Lấy phần chữ mô tả, bỏ mã số D001, F001 ở đầu
                        raw_label = " ".join([str(x) for x in r[:2] if x]).strip().upper().replace("\n", " ")
                        clean_label = re.sub(r'^[A-Z]\d{1,4}[A-Z]?(\.\d+)?\s*', '', raw_label)
                        
                        # Chỉ lấy những hàng có số đo kích thước thực tế (Inches: 3.0 - 100.0)
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(clean_label) > 3:
                            specs[clean_label[:60]] = round(float(np.median(vals)), 2)
                            
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.2, 2.2)).tobytes("png")
        doc.close()
        
        # Nhận diện loại nâng cao
        cat = "ÁO"
        if any(x in str(specs.keys()) for x in ['INSEAM', 'WAIST', 'HIP', 'THIGH', 'KNEE', 'LEG']):
            cat = "QUẦN DÀI LƯNG THƯỜNG"
            if 'ELASTIC' in text.upper() or 'WAISTBAND' in text.upper(): cat = "QUẦN DÀI LƯNG THUN"
            if 0 < specs.get('INSEAM', 0) < 15: cat = "QUẦN SHORT"

        return {"spec": specs, "img": img, "cat": cat, "name": os.path.basename(pdf_path)}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF mới vào kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0); s_text = st.empty()
        for idx, f in enumerate(files):
            p_bar.progress((idx + 1) / len(files))
            s_text.info(f"Đang nạp ({idx+1}/{len(files)}): {f.name}")
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                buf = io.BytesIO()
                img_p.save(buf, format="WEBP", quality=75)
                fname = re.sub(r'[^a-zA-Z0-9]', '_', f.name) + ".webp"
                supabase.storage.from_(BUCKET_NAME).upload(path=fname, file=buf.getvalue(), file_options={"upsert":"true"})
                url = supabase.storage.from_(BUCKET_NAME).get_public_url(fname)
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": vec, "spec_json": d['spec'], "img_url": url, "category": d['cat']}, on_conflict="file_name").execute()
        st.success("🏁 Hoàn tất!"); st.rerun()

# ================= CHÍNH: SO SÁNH FULL THÔNG SỐ =================
st.title("👔 AI Fashion Pro V11.18")
test_file = st.file_uploader("Tải file PDF Test (Đối chứng)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        
        # --- CHỌN MÃ HÀNG ĐỂ SO SÁNH ---
        list_names = [item['file_name'] for item in all_samples]
        selected = st.selectbox("🎯 Chọn mã hàng trong kho (hoặc để AI tự tìm):", ["-- Tự động tìm mẫu tương đồng --"] + list_names)
        
        matches = []
        if selected == "-- Tự động tìm mẫu tương đồng --":
            if all_samples:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
                for i in all_samples:
                    if i.get('vector'):
                        v_db = np.array(i['vector']).reshape(1, -1)
                        # FIX LỖI TYPEERROR DÒNG 137: Lấy chính xác phần tử [0][0]
                        sim_val = cosine_similarity(v_test.reshape(1, -1), v_db)[0][0] * 100
                        if i['category'] == target['cat']: sim_val += 5
                        matches.append({"name": i['file_name'], "sim": sim_val, "url": i['img_url'], "spec": i['spec_json']})
                matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
        else:
            for i in all_samples:
                if i['file_name'] == selected:
                    matches = [{"name": i['file_name'], "sim": 100.0, "url": i['img_url'], "spec": i['spec_json']}]
                    break

        if matches:
            for m in matches:
                with st.expander(f"📌 ĐỐI CHIẾU CHI TIẾT: {m['name']} (Độ giống: {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Bản vẽ Test")
                    with c2: st.image(m['url'], caption="Mẫu Trong Kho")
                    with c3:
                        all_k = sorted(list(set(target['spec'].keys()).union(set(m['spec'].keys()))))
                        comp = []
                        for k in all_k:
                            v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                            diff = round(v_t - v_d, 2)
                            comp.append({"Thông số": k, "Test": v_t, "Kho": v_d, "Lệch": diff})
                        
                        df_res = pd.DataFrame(comp)
                        # Hiển thị bảng so sánh full chữ, số làm tròn 2 chữ số thập phân
                        st.table(df_res.style.format(subset=['Test', 'Kho', 'Lệch'], precision=2).map(lambda x: 'color: red' if abs(x) > 0.25 else 'color: green', subset=['Lệch']))
        else: st.warning("Kho chưa có mẫu phù hợp để so sánh.")
