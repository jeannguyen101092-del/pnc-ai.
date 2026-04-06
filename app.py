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

st.set_page_config(layout="wide", page_title="AI FASHION PRO V15.5", page_icon="🛡️")

# CSS Pro
st.markdown("""<style>.stMetric {background: white; border: 1px solid #ddd; padding: 15px; border-radius: 10px;}</style>""", unsafe_allow_html=True)

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 500

@st.cache_resource
def load_ai():
    # ResNet50 soi chi tiết túi, nắp, lưng thun cực chuẩn
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children() if hasattr(model, 'features') else list(model.children())[:-1]))).eval()

ai_brain = load_ai()

# ================= HÀM TIỆN ÍCH PHÂN TÍCH (PARSER) =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): return ai_brain(tf(img).unsqueeze(0)).flatten().numpy()
    except: return None

def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split(); return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def excel_to_img_pro(file_obj):
    try:
        # Tự động chọn engine cho .xls (xlrd) hoặc .xlsx (openpyxl)
        engine = 'xlrd' if file_obj.name.lower().endswith('.xls') else None
        df = pd.read_excel(file_obj, engine=engine).dropna(how='all', axis=0).fillna("")
        fig, ax = plt.subplots(figsize=(24, len(df.head(60)) * 0.7 + 2)) 
        ax.axis('off')
        ax.table(cellText=df.head(60).values, colLabels=df.columns, loc='center', cellLoc='left').scale(1.2, 3)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=160); plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        st.error(f"Lỗi đọc Excel: {e}")
        return None

def extract_pdf_matrix(pdf_path):
    specs, text = {}, ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text += t
                for tb in p.extract_tables():
                    if not tb or len(tb) < 2: continue
                    for r in tb:
                        if not r or len(r) < 2: continue
                        val = parse_val(r[-1])
                        if val > 0.1:
                            desc = " ".join([str(x or "") for x in r[:-1]]).strip().upper()
                            if len(desc) > 4: specs[re.sub(r'[^A-Z0-9\s/]', '', desc)[:120]] = val
        doc = fitz.open(pdf_path); img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png"); doc.close()
        return {"spec": specs, "img": img, "text": text}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO THÔNG MINH =================
with st.sidebar:
    st.header("📦 HỆ THỐNG KHO V15.5")
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
    st.subheader("🚀 NẠP KHO (PDF + EXCEL)")
    files = st.file_uploader("Kéo thả file vào đây", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'], key=f"up_{st.session_state.up_key}")
    
    if files and st.button("🚀 BẮT ĐẦU NẠP & PHÂN TÍCH AI"):
        groups = {}
        # Thuật toán tìm mã hàng thông minh: Bỏ qua năm 2023, 2024, 2025...
        for f in files:
            # Tìm tất cả chuỗi số dài từ 4 ký tự trở lên
            nums = re.findall(r'\d{4,}', f.name)
            # Lọc bỏ các số năm phổ biến để tìm mã hàng thực sự
            valid_nums = [n for n in nums if n not in ['2023', '2024', '2025', '2026']]
            ma = valid_nums[0] if valid_nums else (nums[0] if nums else "UNK")
            
            ext = "." + f.name.split('.')[-1].lower()
            if ma not in groups: groups[ma] = {}
            groups[ma][ext] = f
        
        for ma, p in groups.items():
            f_pdf = p.get('.pdf')
            f_exl = p.get('.xlsx') or p.get('.xls')
            
            if f_pdf and f_exl:
                with st.spinner(f"AI đang soi mã {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(f_pdf.getbuffer())
                    d, ex_img = extract_pdf_matrix("tmp.pdf"), excel_to_img_pro(f_exl)
                    if d and ex_img:
                        vec = get_vector(d['img']).tolist()
                        try:
                            supabase.storage.from_(BUCKET).upload(f"{ma}_t.webp", d['img'], {"x-upsert": "true"})
                            supabase.storage.from_(BUCKET).upload(f"{ma}_e.webp", ex_img, {"x-upsert": "true"})
                            u_t, u_e = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_t.webp"), supabase.storage.from_(BUCKET).get_public_url(f"{ma}_e.webp")
                            supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t, "excel_img_url": u_e}).execute()
                            st.toast(f"✅ Đã nạp xong mã {ma}")
                        except Exception as e: st.error(f"Lỗi nạp mã {ma}: {e}")
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
            else:
                st.warning(f"⚠️ Mã {ma} thiếu file PDF hoặc Excel!")
        
        st.session_state.up_key += 1; st.rerun()

# ================= MAIN DASHBOARD =================
st.title("🛡️ AI FASHION PRO - SO SÁNH THÔNG MINH")
test_pdf = st.file_uploader("1. Tải PDF Test (File cần kiểm tra)", type="pdf")
target = st.session_state.target

if test_pdf:
    with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
    data_test = extract_pdf_matrix("test.pdf")
    if data_test:
        if st.button("🤖 AI: TỰ ĐỘNG TÌM MÃ GIỐNG NHẤT"):
            test_vec = get_vector(data_test['img'])
            best_sim, best_s = -1, None
            for s in samples:
                if s.get('vector'):
                    sim = cosine_similarity([test_vec], [np.array(s['vector'])])
                    if sim > best_sim: best_sim, best_s = sim, s
            st.session_state.target = best_s; st.rerun()

        col_t, col_k, col_res = st.columns([1, 1, 1.5])
        with col_t:
            st.image(data_test['img'], caption="🖼️ ẢNH PDF ĐANG TEST", use_container_width=True)
        with col_k:
            if target:
                st.image(target['img_url'], caption=f"📁 ẢNH KHO ({target['file_name']})", use_container_width=True)
                st.image(target['excel_img_url'], caption="📊 ĐỊNH MỨC KHO", use_container_width=True)
            else: st.warning("👈 Chọn mã ở Sidebar hoặc bấm nút AI")

        with col_res:
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
                if not df.empty:
                    st.dataframe(df.style.map(lambda x: 'background-color: #ffcccc' if x == "❌ SAI" else ('background-color: #ccffcc' if x == "✅ OK" else ''), subset=['Trạng thái']), use_container_width=True, height=600)
                    output = io.BytesIO()
                    df.to_excel(output, index=False); st.download_button("📥 XUẤT EXCEL SO SÁNH", output.getvalue(), f"Result_{target['file_name']}.xlsx")
