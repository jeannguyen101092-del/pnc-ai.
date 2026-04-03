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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.9", page_icon="👔")

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
    if any(k in txt for k in ['PANT', 'CARGO', 'TROUSER', 'JOGGER']) or length >= 25 or inseam >= 15:
        if any(k in txt for k in ['ELASTIC', 'WAISTBAND', 'THUN', 'RIB']):
            return "QUẦN DÀI LƯNG THUN"
        return "QUẦN DÀI LƯNG THƯỜNG"
    if 0 < length <= 23 or 0 < inseam <= 13 or 'SHORT' in txt:
        return "QUẦN SHORT"
    return "ÁO"

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
                        key_found = None
                        for k in ['INSEAM','WAIST','HIP','LENGTH','OUTSEAM','SLEEVE','SHOULDER','CHEST']:
                            if k in txt_r: key_found = k; break
                        if key_found:
                            vals = [parse_val(x) for x in r if x]
                            valid_vals = [v for v in vals if v >= 4] # Bỏ qua dung sai nhỏ
                            if valid_vals: specs[key_found] = round(float(max(valid_vals)), 2)
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img_bytes, "cat": classify_logic(specs, text, os.path.basename(pdf_path)), "name": os.path.basename(pdf_path)}
    except: return None

# ================= SIDEBAR & NẠP KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_all = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector").execute()
        all_samples = db_all.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []
        st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF vào kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                # Nén WebP
                img_pil = Image.open(io.BytesIO(d['img'])).convert("RGB")
                buf = io.BytesIO()
                img_pil.save(buf, format="WEBP", quality=75)
                img_url = (supabase.storage.from_(BUCKET_NAME).upload(path=f.name+".webp", file=buf.getvalue(), file_options={"upsert":"true"}) and supabase.storage.from_(BUCKET_NAME).get_public_url(f.name+".webp"))
                
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): vec = ai_brain(tf(img_pil).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({"file_name": f.name, "vector": vec, "spec_json": d['spec'], "img_url": img_url, "category": d['cat']}, on_conflict="file_name").execute()
        st.success("🏁 Nạp xong!")
        st.rerun()

# ================= CHÍNH: SO SÁNH NÂNG CAO =================
st.title("👔 AI Fashion Pro V11.9")
test_file = st.file_uploader("Tải file PDF Test (Đối chứng)", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        
        # --- TÍNH NĂNG CHỌN MÃ HÀNG THỦ CÔNG ---
        st.divider()
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            list_names = [item['file_name'] for item in all_samples]
            selected_name = st.selectbox("🎯 Chọn mã hàng cụ thể trong kho để so sánh (hoặc để trống để AI tự tìm):", ["-- Tự động tìm mẫu tương đồng --"] + list_names)
        
        target_matches = []
        
        if selected_name == "-- Tự động tìm mẫu tương đồng --":
            # AI Tự tìm dựa trên Vector và Category
            if all_samples:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
                
                for item in all_samples:
                    if item.get('vector'):
                        sim = float(cosine_similarity(v_test.reshape(1,-1), np.array(item['vector']).reshape(1,-1))) * 100
                        # Ưu tiên mẫu cùng loại
                        if item['category'] == target['cat']: sim += 10 
                        target_matches.append({"name": item['file_name'], "sim": sim, "url": item['img_url'], "spec": item['spec_json']})
                target_matches = sorted(target_matches, key=lambda x: x['sim'], reverse=True)[:3]
        else:
            # Lấy đúng mã người dùng chọn
            for item in all_samples:
                if item['file_name'] == selected_name:
                    target_matches = [{"name": item['file_name'], "sim": 100, "url": item['img_url'], "spec": item['spec_json']}]
                    break

        # --- HIỂN THỊ SO SÁNH ---
        if target_matches:
            # Nút xuất Excel cho các mẫu đang hiện
            export_data = []
            for m in target_matches:
                for k in set(target['spec'].keys()).union(set(m['spec'].keys())):
                    v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                    export_data.append({"Mẫu so sánh": m['name'], "Thông số": k, "Test": v_t, "Kho": v_d, "Lệch": round(v_t-v_d, 2)})
            
            df_ex = pd.DataFrame(export_data)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as wr: df_ex.to_excel(wr, index=False)
            st.download_button("📥 TẢI BÁO CÁO ĐỐI CHIẾU EXCEL", data=buf.getvalue(), file_name=f"SoSanh_{target['name']}.xlsx")

            for m in target_matches:
                with st.expander(f"📌 ĐỐI CHIẾU VỚI: {m['name']} (Độ tương đồng AI: {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1: st.image(target['img'], caption="Ảnh File Test")
                    with c2: st.image(m['url'], caption="Ảnh trong Kho")
                    with c3:
                        comp = []
                        all_keys = sorted(list(set(target['spec'].keys()).union(set(m['spec'].keys()))))
                        for k in all_keys:
                            v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                            diff = round(v_t - v_d, 2)
                            comp.append({"Thông số": k, "File Test": v_t, "Trong Kho": v_d, "Chênh lệch": diff})
                        
                        # Hiển thị bảng với màu sắc cảnh báo lệch
                        df_show = pd.DataFrame(comp)
                        def color_diff(val):
                            color = 'red' if abs(val) > 0.25 else 'green'
                            return f'color: {color}'
                        st.table(df_show.style.applymap(color_diff, subset=['Chênh lệch']))
        else:
            st.warning("Kho hàng trống hoặc không tìm thấy mẫu phù hợp.")
