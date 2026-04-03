import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from difflib import SequenceMatcher

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.29", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= TRÍCH XUẤT DỮ LIỆU & PHÂN LOẠI KHẮT KHE =================
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
    # Lấy Length lớn nhất tìm được
    length = 0
    for k, v in specs.items():
        if 'LENGTH' in k or 'OUTSEAM' in k: length = max(length, v)

    # 1. BỘ LỌC QUẦN SHORT (Ưu tiên hàng đầu nếu ngắn)
    if 'SHORT' in txt or (0 < length < 24) or (0 < inseam < 14):
        return "QUẦN SHORT"

    # 2. BỘ LỌC QUẦN DÀI (Khi đã loại trừ Short)
    if any(k in txt for k in ['PANT', 'CARGO', 'TROUSER', 'JOGGER']) or length >= 24 or inseam >= 14:
        if any(k in txt for k in ['ELASTIC', 'RIB WAIST', 'THUN']):
            return "QUẦN DÀI LƯNG THUN"
        return "QUẦN DÀI LƯNG THƯỜNG"

    return "ÁO / KHÁC"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    content_str = str(tb).upper()
                    if any(x in content_str for x in ['FABRIC', 'MATERIAL', 'BOM']): continue
                    for r in tb:
                        if not r or len(r) < 2: continue
                        label = " ".join([str(x) for x in r[:2] if x]).strip().upper().replace("\n", " ")
                        if any(x in label for x in ['DESCRIPTION', 'TOLERANCE', 'PAGE', 'DATE']): continue
                        label = re.sub(r'^[A-Z]\d{1,4}.*?\s', '', label) # Xóa mã D001
                        # Lấy số đầu tiên tìm được (Size chuẩn)
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(label) > 3:
                            specs[label[:100]] = round(float(vals[0]), 2)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= SIDEBAR & QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF mới", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                buf = io.BytesIO(); img_p.save(buf, format="WEBP", quality=60)
                fname = re.sub(r'[^a-zA-Z0-9]', '_', f.name) + ".webp"
                supabase.storage.from_(BUCKET_NAME).upload(path=fname, file=buf.getvalue(), file_options={"upsert":"true"})
                url = supabase.storage.from_(BUCKET_NAME).get_public_url(fname)
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": vec, "spec_json": d['spec'], "img_url": url, "category": d['cat']}, on_conflict="file_name").execute()
            if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
            gc.collect()
        st.rerun()

# ================= CHÍNH: SO SÁNH (CHỦNG LOẠI -> HÌNH ẢNH -> THÔNG SỐ) =================
st.title("👔 AI Fashion Pro V11.29")
test_file = st.file_uploader("Tải file PDF Test đối chiếu", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện chủng loại: **{target['cat']}**")
        
        # 1. Lọc danh sách: CHỈ lấy những mẫu cùng Category trong kho
        same_cat_samples = [i for i in all_samples if i['category'] == target['cat']]
        
        if not same_cat_samples:
            st.warning(f"⚠️ Không tìm thấy mẫu nào thuộc loại '{target['cat']}' trong kho để so sánh.")
        else:
            # 2. So sánh hình ảnh (Vector) trong danh sách đã lọc
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
            
            matches = []
            for item in same_cat_samples:
                if item.get('vector'):
                    v_db = np.array(item['vector']).reshape(1, -1)
                    sim_val = float(cosine_similarity(v_test.reshape(1, -1), v_db)) * 100
                    matches.append({"name": item['file_name'], "sim": sim_val, "url": item['img_url'], "spec": item['spec_json']})
            
            # Sắp xếp: Thằng nào hình giống nhất lên đầu
            matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]

            # 3. Hiển thị bảng so sánh
            for m in matches:
                with st.expander(f"📌 ĐỐI CHIẾU: {m['name']} (Độ giống hình ảnh: {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Bản vẽ Test")
                    with c2: st.image(m['url'], caption="Ảnh mẫu trong kho")
                    with c3:
                        comp_list = []
                        test_specs, db_specs = target['spec'], m['spec']
                        used_db_keys = set()
                        for kt, vt in test_specs.items():
                            match_key = next((kd for kd in db_specs.keys() if SequenceMatcher(None, kt, kd).ratio() > 0.8), None)
                            if match_key:
                                vd = db_specs[match_key]; used_db_keys.add(match_key)
                                comp_list.append({"Thông số": kt, "Test": vt, "Kho": vd, "Lệch": round(vt - vd, 2)})
                            else:
                                comp_list.append({"Thông số": kt, "Test": vt, "Kho": 0.0, "Lệch": vt})
                        
                        df_res = pd.DataFrame(comp_list)
                        st.table(df_res.style.format(subset=['Test', 'Kho', 'Lệch'], precision=2).map(lambda x: 'color: red' if abs(x) > 0.1 else 'color: green', subset=['Lệch']))
    gc.collect()
