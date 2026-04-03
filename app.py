import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (ĐIỀN THÔNG TIN CỦA BẠN) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.11", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM TIỆN ÍCH PDF & PHÂN LOẠI =================
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
    
    # 1. Nhóm Quần
    if any(k in txt for k in ['PANT', 'CARGO', 'TROUSER']) or length >= 25 or inseam >= 15:
        # CHỈ nhận diện Lưng Thun nếu có từ khóa cực kỳ cụ thể về cạp quần
        if any(k in txt for k in ['ELASTIC WAIST', 'RIB WAIST', 'FULL ELASTIC', 'LƯNG THUN']):
            return "QUẦN DÀI LƯNG THUN"
        # Mặc định là Lưng Thường
        return "QUẦN DÀI LƯNG THƯỜNG"
    
    if 0 < length <= 23 or 0 < inseam <= 13 or 'SHORT' in txt: return "QUẦN SHORT"
    
    # 2. Nhóm Áo
    if any(k in txt for k in ['VEST', 'BLAZER', 'JACKET']): return "ÁO VEST / JACKET"
    sleeve = specs.get('SLEEVE', 0)
    if sleeve >= 20: return "ÁO DÀI TAY"
    if 0 < sleeve <= 12: return "ÁO NGẮN TAY"
    
    return "HÀNG KHÁC"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    for r in tb:
                        if not r: continue
                        txt_r = " | ".join([str(x) for x in r if x]).upper()
                        key_f = None
                        for k in ['INSEAM','WAIST','HIP','LENGTH','OUTSEAM','SLEEVE','SHOULDER','CHEST']:
                            if k in txt_r: key_f = k; break
                        if key_f:
                            vals = [parse_val(x) for x in r if x]
                            valid = [v for v in vals if v >= 4] # Bỏ qua dung sai nhỏ
                            if valid: specs[key_f] = round(float(max(valid)), 2)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path)), "name": os.path.basename(pdf_path)}
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
    files = st.file_uploader("Nạp PDF mới vào kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
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
        st.success("🏁 Nạp kho thành công!")
        st.rerun()

    st.divider()
    st.write("🗑️ Xóa mã hàng")
    del_n = st.selectbox("Chọn mã cần xóa:", ["-- Chọn mã --"] + [i['file_name'] for i in all_samples])
    if del_n != "-- Chọn mã --" and st.button("❌ XÓA VĨNH VIỄN"):
        supabase.table("ai_data").delete().eq("file_name", del_n).execute()
        st.rerun()

# ================= CHÍNH: SO SÁNH =================
st.title("👔 AI Fashion Pro V11.11")
test_file = st.file_uploader("Tải file PDF Test (Đối chứng)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        
        # CHỌN MÃ HÀNG ĐỂ SO SÁNH
        list_n = [item['file_name'] for item in all_samples]
        sel = st.selectbox("🎯 Chọn mã hàng cụ thể (hoặc AI tự tìm mẫu tương đồng):", ["-- Tự động tìm mẫu tương đồng --"] + list_n)
        
        matches = []
        if sel == "-- Tự động tìm mẫu tương đồng --":
            if all_samples:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
                for i in all_samples:
                    if i.get('vector'):
                        v_db = np.array(i['vector']).reshape(1, -1)
                        # Sửa lỗi so sánh AI dòng 117
                        sim_val = cosine_similarity(v_test.reshape(1, -1), v_db)[0][0] * 100
                        if i['category'] == target['cat']: sim_val += 5
                        matches.append({"name": i['file_name'], "sim": sim_val, "url": i['img_url'], "spec": i['spec_json']})
                matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
        else:
            for i in all_samples:
                if i['file_name'] == sel:
                    matches = [{"name": i['file_name'], "sim": 100.0, "url": i['img_url'], "spec": i['spec_json']}]
                    break

        if matches:
            for m in matches:
                with st.expander(f"📌 ĐỐI CHIẾU: {m['name']} (Độ tương đồng: {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns()
                    with c1: st.image(target['img'], caption="File Test")
                    with c2: st.image(m['url'], caption="Mẫu Kho")
                    with c3:
                        comp = []
                        all_k = sorted(list(set(target['spec'].keys()).union(set(m['spec'].keys()))))
                        for k in all_k:
                            v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                            diff = round(v_t - v_d, 2)
                            comp.append({"Thông số": k, "Test": v_t, "Kho": v_d, "Lệch": diff})
                        df = pd.DataFrame(comp)
                        st.table(df.style.applymap(lambda x: 'color: red' if abs(x) > 0.25 else 'color: green', subset=['Lệch']))
