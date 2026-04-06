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

# ================= CONFIG (Thay bằng thông tin của bạn) =================
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

# ================= HÀM CHỤP ẢNH EXCEL ĐỊNH MỨC =================
def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).fillna("")
        df_display = df.head(30)
        fig, ax = plt.subplots(figsize=(10, len(df_display) * 0.5))
        ax.axis('off')
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        return buf.getvalue()
    except: return None

# ================= TRÍCH XUẤT DỮ LIỆU PDF =================
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
    if 'SHORT' in txt or (0 < length < 24) or (0 < inseam < 14): return "QUẦN SHORT"
    if any(k in txt for k in ['PANT', 'CARGO', 'TROUSER', 'JOGGER']) or length >= 24 or inseam >= 14:
        if any(k in txt for k in ['ELASTIC', 'RIB WAIST', 'THUN']): return "QUẦN DÀI LƯNG THUN"
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
                    for r in tb:
                        if not r or len(r) < 2: continue
                        label = " ".join([str(x) for x in r[:2] if x]).strip().upper()
                        vals = [parse_val(x) for x in r[1:] if 3.0 <= parse_val(x) <= 100.0]
                        if vals and len(label) > 3: specs[label[:100]] = round(float(vals[0]), 2)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": classify_logic(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= SIDEBAR & NẠP KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("*").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    except: 
        all_samples = []; st.metric("Tổng mẫu trong kho", "0 mẫu")
    
    st.divider()
    files = st.file_uploader("Nạp PDF & Excel (Cùng mã số đầu)", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])
    
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        groups = {}
        for f in files:
            m = re.search(r'^\d+', f.name)
            if m:
                ma = m.group()
                ext = os.path.splitext(f.name)[1].lower()
                if ma not in groups: groups[ma] = {}
                groups[ma][ext] = f

        for ma, parts in groups.items():
            f_pdf = parts.get('.pdf')
            f_exl = parts.get('.xlsx') or parts.get('.xls')
            if f_pdf and f_exl:
                with st.spinner(f"Đang nạp mã: {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(f_pdf.getbuffer())
                    d = get_data("tmp.pdf")
                    exl_img = excel_to_img_bytes(f_exl)
                    if d and exl_img:
                        img_p = Image.open(io.BytesIO(d['img'])).convert("RGB")
                        buf = io.BytesIO(); img_p.save(buf, format="WEBP")
                        supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_t.webp", buf.getvalue(), {"upsert":"true"})
                        url_t = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_t.webp")
                        supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_e.webp", exl_img, {"upsert":"true"})
                        url_e = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_e.webp")
                        
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                        with torch.no_grad(): vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                        
                        supabase.table("ai_data").upsert({
                            "file_name": ma, "vector": vec, "spec_json": d['spec'], 
                            "img_url": url_t, "excel_img_url": url_e, "category": d['cat']
                        }, on_conflict="file_name").execute()
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.rerun()

# ================= CHÍNH: SO SÁNH =================
st.title("👔 AI Fashion Pro V11.29")
test_file = st.file_uploader("Tải PDF Test", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")
    if target:
        st.subheader(f"Nhận diện chủng loại: **{target['cat']}**")
        same_cat = [i for i in all_samples if i['category'] == target['cat']]
        
        if not same_cat:
            st.warning(f"⚠️ Không tìm thấy mẫu cùng loại '{target['cat']}' trong kho.")
        else:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().detach().numpy()
            
            matches = []
            for item in same_cat:
                if item.get('vector'):
                    try:
                        v_raw = item['vector']
                        if isinstance(v_raw, str): 
                            v_raw = [float(x) for x in v_raw.strip('[]').split(',')]
                        v_db = np.array(v_raw, dtype=np.float32).reshape(1, -1)
                        # SỬA LỖI TẠI ĐÂY: Thêm [0][0] để lấy giá trị số từ mảng kết quả
                        sim = float(cosine_similarity(v_test.reshape(1, -1), v_db)[0][0]) * 100
                        matches.append(item | {"sim": sim})
                    except: continue
            
            for m in sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]:
                with st.expander(f"📌 ĐỐI CHIẾU: {m['file_name']} ({m['sim']:.1f}%)", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1.8])
                    with c1: st.image(target['img'], caption="Bản vẽ Test")
                    with c2: 
                        st.image(m['img_url'], caption="Ảnh mẫu trong kho")
                        if m.get('excel_img_url'): st.image(m['excel_img_url'], caption="Định mức (Excel)")
                    with c3:
                        res = []
                        test_specs, db_specs = target['spec'], m['spec_json']
                        for kt, vt in test_specs.items():
                            mk = next((k for k in db_specs.keys() if SequenceMatcher(None, kt, k).ratio() > 0.8), None)
                            vd = db_specs[mk] if mk else 0.0
                            res.append({"Thông số": kt, "Test": vt, "Kho": vd, "Lệch": round(vt - vd, 2)})
                        
                        df_res = pd.DataFrame(res)
                        st.table(df_res)
                        
                        # XUẤT EXCEL
                        out = io.BytesIO()
                        with pd.ExcelWriter(out, engine='xlsxwriter') as wr: 
                            df_res.to_excel(wr, index=False, sheet_name='Comparison')
                        st.download_button(
                            label=f"📥 XUẤT EXCEL KẾT QUẢ: {m['file_name']}",
                            data=out.getvalue(),
                            file_name=f"SoSanh_{m['file_name']}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if os.path.exists("test.pdf"): os.remove("test.pdf")
gc.collect()
