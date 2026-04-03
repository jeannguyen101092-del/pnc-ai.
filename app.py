import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from difflib import SequenceMatcher # Thêm thư viện so sánh chữ thông minh

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.27", page_icon="👔")

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= XỬ LÝ DỮ LIỆU PDF CHUẨN HÓA =================
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
    # Nếu có thông số Gối (KNEE) hoặc Inseam dài -> QUẦN DÀI
    inseam = specs.get('INSEAM', 0)
    has_knee = any('KNEE' in k.upper() for k in specs.keys())
    
    if inseam >= 20 or has_knee: return "QUẦN DÀI"
    if 'SHORT' in txt or specs.get('INSEAM', 0) < 15: return "QUẦN SHORT"
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
                        # BỘ LỌC TIÊU ĐỀ RÁC
                        if any(x in label for x in ['DESCRIPTION', 'SAMPLE', 'HEADER', 'PAGE', 'DATE', 'TOLERANCE']): continue
                        label = re.sub(r'^[A-Z]\d{1,4}.*?\s', '', label)
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(label) > 3:
                            specs[label[:100]] = round(float(vals[0]), 2)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
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
        st.rerun()

# ================= CHÍNH: SO SÁNH THÔNG MINH =================
st.title("👔 AI Fashion Pro V11.27")
test_file = st.file_uploader("Tải file PDF Test đối chiếu", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        list_names = [item['file_name'] for item in all_samples]
        selected = st.selectbox("🎯 Chọn mã hàng trong kho:", ["-- Tự động tìm mẫu tương đồng --"] + list_names)
        
        matches = []
        if selected == "-- Tự động tìm mẫu tương đồng --":
            if all_samples:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
                for i in all_samples:
                    if i.get('vector'):
                        v_db = np.array(i['vector']).reshape(1, -1)
                        sim_val = float(cosine_similarity(v_test.reshape(1, -1), v_db)[0][0]) * 100
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
                with st.expander(f"📌 ĐỐI CHIẾU: {m['name']} (Độ tương đồng: {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Bản vẽ Test")
                    with c2: st.image(m['url'], caption="Ảnh Kho")
                    with c3:
                        # --- LOGIC KHỚP THÔNG SỐ FUZZY (MỚI) ---
                        comp_list = []
                        test_specs = target['spec']
                        db_specs = m['spec']
                        used_db_keys = set()

                        for kt, vt in test_specs.items():
                            # Tìm key giống nhất trong kho (>80% similarity)
                            best_match = None; max_ratio = 0
                            for kd in db_specs.keys():
                                ratio = SequenceMatcher(None, kt, kd).ratio()
                                if ratio > max_ratio and ratio > 0.8:
                                    max_ratio = ratio; best_match = kd
                            
                            if best_match:
                                vd = db_specs[best_match]; used_db_keys.add(best_match)
                                diff = round(vt - vd, 2)
                                comp_list.append({"Thông số": kt, "Test": vt, "Kho": vd, "Lệch": diff})
                            else:
                                comp_list.append({"Thông số": kt, "Test": vt, "Kho": 0.0, "Lệch": vt})

                        for kd, vd in db_specs.items():
                            if kd not in used_db_keys:
                                comp_list.append({"Thông số": kd, "Test": 0.0, "Kho": vd, "Lệch": -vd})
                        
                        df_res = pd.DataFrame(comp_list)
                        st.table(df_res.style.format(subset=['Test', 'Kho', 'Lệch'], precision=2).map(lambda x: 'color: red' if abs(x) > 0.05 else 'color: green', subset=['Lệch']))
    gc.collect()
