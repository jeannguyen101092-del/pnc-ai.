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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.5", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM TIỆN ÍCH =================
def compress_to_webp(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=70, method=6)
    return buf.getvalue()

def upload_to_storage(img_bytes, filename):
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename) + ".webp"
        supabase.storage.from_(BUCKET_NAME).upload(
            path=clean_name, file=img_bytes,
            file_options={"content-type": "image/webp", "upsert": "true"}
        )
        return supabase.storage.from_(BUCKET_NAME).get_public_url(clean_name)
    except: return None

def parse_val(t):
    try:
        # Xử lý phân số phức tạp như "30 1/2" hoặc "1/4"
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def classify_logic(specs, text, name):
    txt = (text + name).upper()
    inseam = specs.get('INSEAM', 0)
    length = specs.get('LENGTH', 0)

    # Nếu có từ khóa Pant/Cargo hoặc chiều dài thực sự lớn
    if 'CARGO' in txt or 'PANT' in txt or length > 30 or inseam > 20:
        return "QUẦN DÀI"
    if 0 < length <= 22 or 0 < inseam <= 15:
        return "QUẦN SHORT"
    if 'SHORT' in txt: return "QUẦN SHORT"
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
                        # Ưu tiên các từ khóa thông số chính
                        for k in ['WAIST','HIP','INSEAM','LENGTH','THIGH','KNEE','LEG OPEN','CHEST','SHOULDER']:
                            if k in txt_r: 
                                key_found = k
                                break
                        
                        if key_found:
                            # LẤY SỐ: Bỏ qua các số nhỏ < 3 (thường là dung sai +/-)
                            vals = [parse_val(x) for x in r if x]
                            valid_vals = [v for v in vals if v >= 3] # Chỉ lấy thông số thực > 3 inch
                            
                            if valid_vals:
                                # Lấy giá trị lớn nhất (thường là size lớn nhất hoặc base size)
                                specs[key_found] = round(float(max(valid_vals)), 2)
        
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img_bytes, "cat": classify_logic(specs, text, os.path.basename(pdf_path)), "name": os.path.basename(pdf_path)}
    except: return None

# ================= GIAO DIỆN =================
st.title("👔 AI Fashion Pro V11.5 - Fix Specification")

with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        res_count = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{res_count.count} mẫu")
    except: st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF vào kho", accept_multiple_files=True)
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            if d:
                img_webp = compress_to_webp(d['img'])
                img_url = upload_to_storage(img_webp, f.name)
                if img_url:
                    tf = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        vec = ai_brain(tf(Image.open(io.BytesIO(img_webp))).unsqueeze(0)).flatten().numpy().tolist()
                    supabase.table("ai_data").upsert({
                        "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                        "img_url": img_url, "category": d['cat']
                    }, on_conflict="file_name").execute()
        st.success("🏁 Nạp kho thành công!")
        st.rerun()

# ================= CHÍNH: SO SÁNH =================
test_file = st.file_uploader("Tải file PDF Test", type="pdf")
if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        if db.data:
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
            
            matches = []
            for item in db.data:
                if item.get('vector'):
                    v_db = np.array(item['vector'], dtype=np.float32)
                    sim = float(cosine_similarity(v_test.reshape(1,-1), v_db.reshape(1,-1))) * 100
                    matches.append({"name": item['file_name'], "sim": sim, "url": item['img_url'], "spec": item['spec_json']})
            
            top_matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
            
            # Xuất Excel
            export_data = []
            for m in top_matches:
                keys = set(target['spec'].keys()).union(set(m['spec'].keys()))
                for k in sorted(keys):
                    v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                    export_data.append({"Mẫu": m['name'], "Thông số": k, "Test": v_t, "Kho": v_d, "Lệch": round(v_t-v_d, 2)})
            
            if export_data:
                df_ex = pd.DataFrame(export_data)
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as wr: df_ex.to_excel(wr, index=False)
                st.download_button("📥 TẢI BÁO CÁO EXCEL", data=buf.getvalue(), file_name="So_Sanh_Fashion.xlsx")

            for m in top_matches:
                with st.expander(f"Mẫu: {m['name']} (Khớp {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns()
                    with c1: st.image(target['img'], caption="Test")
                    with c2: st.image(m['url'], caption="Kho")
                    with c3:
                        comp = []
                        keys = set(target['spec'].keys()).union(set(m['spec'].keys()))
                        for k in sorted(keys):
                            v_t, v_d = target['spec'].get(k, 0), m['spec'].get(k, 0)
                            comp.append({"Thông số": k, "Test": v_t, "Kho": v_d, "Lệch": round(v_t-v_d, 2)})
                        st.table(pd.DataFrame(comp))
