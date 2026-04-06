import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG (Giữ nguyên) =================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
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

# ================= HÀM CHỤP ẢNH EXCEL =================
def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).fillna("")
        df = df.head(30) # Chụp 30 dòng đầu
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.4))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        return buf.getvalue()
    except:
        return None

# ================= LOGIC TRÍCH XUẤT (Giữ nguyên của bạn) =================
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

    if 'SHORT' in txt or (0 < length < 24) or (0 < inseam < 14):
        return "QUẦN SHORT"
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
                        label = re.sub(r'^[A-Z]\d{1,4}.*?\s', '', label)
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(label) > 3:
                            specs[label[:100]] = round(float(vals[0]), 2)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= SIDEBAR & QUẢN LÝ KHO (Đã cập nhật logic nạp song song) =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("file_name, category, spec_json, img_url, vector, excel_img_url").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF & Excel (Cùng tên file)", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])
    
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        # Phân loại file theo tên gốc (không bao gồm đuôi)
        groups = {}
        for f in files:
            name_part = os.path.splitext(f.name)[0]
            ext_part = os.path.splitext(f.name)[1].lower()
            if name_part not in groups: groups[name_part] = {}
            groups[name_part][ext_part] = f

        for name, parts in groups.items():
            # ĐIỀU KIỆN: Có PDF và có EXCEL mới làm
            pdf_file = parts.get('.pdf')
            exl_file = parts.get('.xlsx') or parts.get('.xls')

            if pdf_file and exl_file:
                with st.spinner(f"Đang xử lý mã: {name}..."):
                    with open("tmp.pdf", "wb") as t: t.write(pdf_file.getbuffer())
                    d = get_data("tmp.pdf")
                    exl_img_data = excel_to_img_bytes(exl_file)

                    if d and exl_img_data:
                        # 1. Lưu ảnh PDF
                        img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                        buf_pdf = io.BytesIO(); img_p.save(buf_pdf, format="WEBP", quality=60)
                        fname_pdf = f"{name}_tech.webp"
                        supabase.storage.from_(BUCKET_NAME).upload(fname_pdf, buf_pdf.getvalue(), {"upsert":"true"})
                        url_pdf = supabase.storage.from_(BUCKET_NAME).get_public_url(fname_pdf)

                        # 2. Lưu ảnh Excel định mức
                        fname_exl = f"{name}_dm.webp"
                        supabase.storage.from_(BUCKET_NAME).upload(fname_exl, exl_img_data, {"upsert":"true"})
                        url_exl = supabase.storage.from_(BUCKET_NAME).get_public_url(fname_exl)

                        # 3. Vector ảnh
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                        with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                        
                        # 4. Lưu DB
                        supabase.table("ai_data").upsert({
                            "file_name": name, 
                            "vector": vec, 
                            "spec_json": d['spec'], 
                            "img_url": url_pdf, 
                            "excel_img_url": url_exl,
                            "category": d['cat']
                        }, on_conflict="file_name").execute()
                    
                    if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
            else:
                st.warning(f"Bỏ qua {name}: Thiếu file PDF hoặc Excel tương ứng.")
        st.rerun()

# ================= CHÍNH: SO SÁNH =================
st.title("👔 AI Fashion Pro V11.29")
test_file = st.file_uploader("Tải file PDF Test đối chiếu", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện chủng loại: **{target['cat']}**")
        same_cat_samples = [i for i in all_samples if i['category'] == target['cat']]
        
        if not same_cat_samples:
            st.warning(f"⚠️ Không tìm thấy mẫu cùng loại '{target['cat']}'")
        else:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            with torch.no_grad(): v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()
            
            matches = []
            for item in same_cat_samples:
                if item.get('vector'):
                    v_db = np.array(item['vector']).reshape(1, -1)
                    sim_val = float(cosine_similarity(v_test.reshape(1, -1), v_db)) * 100
                    matches.append(item | {"sim": sim_val})
            
            matches = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]

            for m in matches:
                with st.expander(f"📌 ĐỐI CHIẾU: {m['file_name']} (Độ giống: {m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Bản vẽ Test")
                    with c2: 
                        st.image(m['img_url'], caption="Ảnh mẫu trong kho")
                        if m.get('excel_img_url'):
                            st.image(m['excel_img_url'], caption="Định mức mã hàng")
                    with c3:
                        comp_list = []
                        test_specs, db_specs = target['spec'], m['spec_json']
                        for kt, vt in test_specs.items():
                            match_key = next((kd for kd in db_specs.keys() if SequenceMatcher(None, kt, kd).ratio() > 0.8), None)
                            if match_key:
                                vd = db_specs[match_key]
                                comp_list.append({"Thông số": kt, "Test": vt, "Kho": vd, "Lệch": round(vt - vd, 2)})
                            else:
                                comp_list.append({"Thông số": kt, "Test": vt, "Kho": 0.0, "Lệch": vt})
                        st.table(pd.DataFrame(comp_list))
