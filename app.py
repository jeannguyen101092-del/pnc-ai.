import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, datetime, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG (Thay URL và KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI FASHION PRO V15.1", page_icon="🛡️")

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 200

@st.cache_resource
def load_ai():
    # Sử dụng ResNet50 để soi kỹ chi tiết túi, dáng áo quần
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM HỖ TRỢ SOI KỸ CHI TIẾT =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): return ai_brain(tf(img).unsqueeze(0)).flatten().numpy()
    except: return None

def identify_fashion_features(text):
    """AI soi văn bản để nhận diện túi, thun, tay áo"""
    txt = str(text).upper()
    feats = []
    if 'CARGO' in txt: feats.append("Túi Hộp (Cargo)")
    if 'ELASTIC' in txt: feats.append("Lưng Thun")
    if 'SLANT' in txt: feats.append("Túi Xéo")
    if 'LONG SLEEVE' in txt: feats.append("Dài Tay")
    if 'SKORT' in txt: feats.append("Quần Váy")
    return feats

def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found
        if ' ' in v:
            p = v.split(); return float(p) + eval(p)
        return eval(v) if '/' in v else float(v)
    except: return 0

def excel_to_img(file_obj):
    try:
        engine = 'xlrd' if file_obj.name.endswith('.xls') else None
        df = pd.read_excel(file_obj, engine=engine).dropna(how='all', axis=0).fillna("")
        fig, ax = plt.subplots(figsize=(22, len(df.head(60)) * 0.6 + 2)) 
        ax.axis('off')
        ax.table(cellText=df.head(60).values, colLabels=df.columns, loc='center', cellLoc='left').scale(1.2, 2.8)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=160); plt.close(fig)
        return buf.getvalue()
    except: return None

def get_data(pdf_path):
    specs, text = {}, ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text += t
                # Quét bảng thông số
                for tb in p.extract_tables():
                    if not tb or len(tb) < 2: continue
                    # Logic tìm bảng Specs Pro
                    for r in tb:
                        if not r or len(r) < 2: continue
                        val = parse_val(r[-1])
                        if val > 0.1:
                            desc = " ".join([str(x or "") for x in r[:-1]]).strip().upper()
                            if len(desc) > 5: specs[desc[:120]] = val
        doc = fitz.open(pdf_path); img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png"); doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 HỆ THỐNG KHO V15.1")
    # FIX LỖI: Chỉ lấy file_name để tránh lỗi thiếu cột features
    try:
        db_res = supabase.table("ai_data").select("*").execute()
        samples = db_res.data if db_res else []
    except: samples = []
    
    st.metric("Tổng mẫu trong kho", len(samples))
    
    list_ma = [s['file_name'] for s in samples]
    sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU CỐ ĐỊNH", ["-- Chọn --"] + list_ma)
    if sel != "-- Chọn --":
        st.session_state.target = next(s for s in samples if s['file_name'] == sel)

    st.divider()
    files = st.file_uploader("Nạp PDF & Excel mới", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'], key=f"up_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP & PHÂN TÍCH AI"):
        groups = {}
        for f in files:
            nums = re.findall(r'\d{3,}', f.name)
            if nums:
                ma = max(nums, key=len)
                ext = "." + f.name.split('.')[-1].lower()
                if ma not in groups: groups[ma] = {}
                groups[ma][ext] = f
        
        for ma, p in groups.items():
            f_pdf, f_exl = p.get('.pdf'), (p.get('.xlsx') or p.get('.xls'))
            if f_pdf and f_exl:
                with st.spinner(f"Soi mã {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(f_pdf.getbuffer())
                    d, ex_img = get_data("tmp.pdf"), excel_to_img(f_exl)
                    if d and ex_img:
                        vec = get_vector(d['img']).tolist()
                        supabase.storage.from_(BUCKET).upload(f"{ma}_t.webp", d['img'], {"x-upsert": "true"})
                        supabase.storage.from_(BUCKET).upload(f"{ma}_e.webp", ex_img, {"x-upsert": "true"})
                        u_t, u_e = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_t.webp"), supabase.storage.from_(BUCKET).get_public_url(f"{ma}_e.webp")
                        # Lưu dữ liệu an toàn
                        supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t, "excel_img_url": u_e}).execute()
                        st.toast(f"✅ Xong {ma}")
        st.session_state.up_key += 1; st.rerun()

# ================= MAIN DASHBOARD =================
st.title("🛡️ AI FASHION PRO - SO SÁNH THÔNG MINH")
test_pdf = st.file_uploader("1. Tải PDF Test", type="pdf")
target = st.session_state.target

if test_pdf:
    with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
    data_test = get_data("test.pdf")
    if data_test:
        if st.button("🤖 AI: TỰ ĐỘNG NHẬN DIỆN MÃ TƯƠNG ĐỒNG NHẤT"):
            test_vec = get_vector(data_test['img'])
            best_sim, best_s = -1, None
            for s in samples:
                if s.get('vector'):
                    sim = cosine_similarity([test_vec], [np.array(s['vector'])])
                    if sim > best_sim: best_sim, best_s = sim, s
            st.session_state.target = best_s; st.rerun()

        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            st.image(data_test['img'], caption="🖼️ ẢNH TEST", use_container_width=True)
            # Soi chi tiết Test
            for ft in identify_fashion_features(data_test): st.info(f"🔍 {ft}")
        
        with c2:
            if target:
                st.image(target['img_url'], caption=f"📁 KHO ({target['file_name']})", use_container_width=True)
                st.image(target['excel_img_url'], caption="📊 ĐỊNH MỨC KHO", use_container_width=True)
            else: st.warning("👈 Chọn mã ở Sidebar")

        with c3:
            if target:
                st.subheader(f"📊 Đối chiếu: {target['file_name']}")
                rows = []
                for kt, vt in data_test['spec'].items():
                    best_m, high_r = None, 0
                    for kb in target['spec_json'].keys():
                        r = SequenceMatcher(None, kt, kb).ratio()
                        if r > high_r: high_r, best_m = r, kb
                    v_db = target['spec_json'].get(best_m, 0) if high_r > 0.6 else "N/A"
                    diff = round(vt - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"
                    rows.append({"Thông số PDF": kt, "Test": vt, "Kho": v_db, "Chênh lệch": diff, "Trạng thái": "✅ OK" if diff == 0 else "❌ SAI"})
                df = pd.DataFrame(rows)
                st.dataframe(df.style.map(lambda x: 'background-color: #ffcccc' if x == "❌ SAI" else ('background-color: #ccffcc' if x == "✅ OK" else ''), subset=['Trạng thái']), use_container_width=True, height=600)
                output = io.BytesIO()
                df.to_excel(output, index=False); st.download_button("📥 XUẤT EXCEL", output.getvalue(), f"Result_{target['file_name']}.xlsx")
