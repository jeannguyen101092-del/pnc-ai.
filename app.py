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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.25", page_icon="👔")

# ================= AI ENGINE (SIÊU NHẸ) =================
@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM XỬ LÝ DỮ LIỆU PDF =================
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
    length = 0
    for k, v in specs.items():
        if 'LENGTH' in k or 'OUTSEAM' in k: length = max(length, v)
    
    # ƯU TIÊN SỐ ĐO: Nếu ngắn là SHORT ngay
    if (0 < length < 25) or (0 < inseam < 14) or 'SHORT' in txt: return "QUẦN SHORT"
    if any(k in txt for k in ['PANT', 'CARGO', 'TROUSER']) or length >= 25:
        if any(k in txt for k in ['ELASTIC', 'RIB WAIST', 'THUN']): return "QUẦN DÀI LƯNG THUN"
        return "QUẦN DÀI LƯNG THƯỜNG"
    return "ÁO / KHÁC"

 get_data(pdf_path):
    try:
        specdefs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    content = str(tb).upper()
                    if any(x in content for x in ['FABRIC', 'MATERIAL', 'BOM']): continue
                    for r in tb:
                        if not r or len(r) < 2: continue
                        # Làm sạch tên thông số
                        label = " ".join([str(x) for x in r[:2] if x]).strip().upper().replace("\n", " ")
                        label = re.sub(r'^[A-Z]\d{1,4}.*?\s', '', label) # Xóa mã D001...
                        # Lấy số đo thực tế (3-100 inch)
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(label) > 3:
                            specs[label[:100]] = round(float(np.median(vals)), 2)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return Nonedef get_data(pdf_path):
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
                        label = re.sub(r'^[A-Z]\d{1,4}.*?\s', '', label)
                        
                        # Lấy tất cả số đo thực tế
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        
                        if vals and len(label) > 3:
                            # THAY ĐỔI TẠI ĐÂY: Lấy vals[0] (số đầu tiên) thay vì median
                            # Việc này đảm bảo luôn lấy đúng 1 cột size cố định
                            specs[label[:100]] = round(float(vals[0]), 2) 
                            
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None


# ================= SIDEBAR & NẠP KHO =================
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
        p_bar = st.progress(0)
        for idx, f in enumerate(files):
            p_bar.progress((idx + 1) / len(files))
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
        st.success("🏁 Nạp xong!"); st.rerun()

# ================= CHÍNH: SO SÁNH THÔNG MINH =================
st.title("👔 AI Fashion Pro V11.25")
test_file = st.file_uploader("Tải file PDF Test đối chiếu", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện loại: {target['cat']}")
        list_names = [item['file_name'] for item in all_samples]
        selected = st.selectbox("🎯 Chọn mã hàng trong kho (hoặc AI tự tìm):", ["-- Tự động tìm mẫu tương đồng --"] + list_names)
        
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
                with st.expander(f"📌 ĐỐI CHIẾU: {m['name']} (Giống {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Ảnh Test")
                    with c2: st.image(m['url'], caption="Ảnh Kho")
                    with c3:
                        # --- LOGIC KHỚP THÔNG SỐ THÔNG MINH ---
                        comp_list = []
                        test_specs = target['spec']
                        db_specs = m['spec']
                        used_db_keys = set()

                        for k_t, v_t in test_specs.items():
                            # Tìm key tương đồng nhất (bỏ qua dấu cách, ký tự đặc biệt)
                            match_key = next((kd for kd in db_specs.keys() if kd.strip() == k_t.strip() or k_t[:20] in kd), None)
                            if match_key:
                                v_d = db_specs[match_key]
                                used_db_keys.add(match_key)
                                diff = round(v_t - v_d, 2)
                                comp_list.append({"Thông số": k_t, "Test": v_t, "Kho": v_d, "Lệch": diff})
                            else:
                                comp_list.append({"Thông số": k_t, "Test": v_t, "Kho": 0.0, "Lệch": v_t})

                        for k_d, v_d in db_specs.items():
                            if k_d not in used_db_keys:
                                comp_list.append({"Thông số": k_d, "Test": 0.0, "Kho": v_d, "Lệch": -v_d})
                        
                        df_res = pd.DataFrame(comp_list)
                        
                        # Xuất Excel
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df_res.to_excel(writer, index=False)
                        st.download_button(label="📥 Tải Excel Đối Chiếu", data=output.getvalue(), file_name=f"SoSanh_{m['name']}.xlsx")
                        
                        # Hiển thị bảng
                        st.table(df_res.style.format(subset=['Test', 'Kho', 'Lệch'], precision=2).map(lambda x: 'color: red' if abs(x) > 0.25 else 'color: green', subset=['Lệch']))
    gc.collect()
