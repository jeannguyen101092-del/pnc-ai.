import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, datetime, logging
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

st.set_page_config(layout="wide", page_title="AI FASHION PRO V15.8", page_icon="🛡️")

# CSS Pro giao diện tối giản
st.markdown("""<style>.stMetric {background: white; border: 1px solid #ddd; padding: 10px; border-radius: 8px;}</style>""", unsafe_allow_html=True)

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 100

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM XỬ LÝ DỮ LIỆU CHUẨN KỸ THUẬT =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): return ai_brain(tf(img).unsqueeze(0)).flatten().numpy()
    except: return None

def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        nums = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not nums: return 0
        v = nums[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def excel_to_img(file_obj):
    """Sửa lỗi XLS bằng cách ép engine xlrd"""
    try:
        ext = file_obj.name.split('.')[-1].lower()
        engine = 'xlrd' if ext == 'xls' else 'openpyxl'
        df = pd.read_excel(file_obj, engine=engine).dropna(how='all', axis=0).fillna("")
        
        fig, ax = plt.subplots(figsize=(20, len(df.head(60)) * 0.6 + 2))
        ax.axis('off')
        ax.table(cellText=df.head(60).values, colLabels=df.columns, loc='center', cellLoc='left').scale(1, 2.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=160); plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        st.error(f"❌ Lỗi định dạng Excel: {e}. Hãy cài 'pip install xlrd'")
        return None

def extract_pdf(pdf_path):
    specs, text, base_size = {}, "", "8"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text += txt
                m = re.search(r'(?:Base|Sample)\s*Size\s*[:\s]\s*(\w+)', txt, re.I)
                if m: base_size = m.group(1).upper()

                for tb in page.extract_tables():
                    if not tb or len(tb) < 2: continue
                    h_idx, header = -1, []
                    for i, row in enumerate(tb[:10]):
                        row_up = [str(x or "").strip().upper() for x in row]
                        if any(base_size == x or (base_size in x and len(x) < 5) for x in row_up):
                            h_idx, header = i, row_up; break
                    if h_idx != -1:
                        b_idx = next((idx for idx, v in enumerate(header) if base_size in v and len(v) < 6), -1)
                        if b_idx != -1:
                            for r in tb[h_idx + 1:]:
                                desc = " ".join([str(x or "") for x in r[:b_idx]]).strip().upper()
                                val = parse_val(r[b_idx])
                                if val > 0.1 and len(desc) > 5:
                                    specs[re.sub(r'[^A-Z0-9\s/]', '', desc)[:120]] = round(float(val), 3)
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
supabase: Client = create_client(URL, KEY)

with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res else []
    st.metric("Tổng mẫu trong kho", len(samples))
    
    list_ma = [s['file_name'] for s in samples]
    sel = st.selectbox("🎯 CHỌN MÃ ĐỐI CHIẾU", ["-- Chọn mã --"] + list_ma)
    if sel != "-- Chọn mã --": st.session_state.target = next(s for s in samples if s['file_name'] == sel)

    st.divider()
    up_files = st.file_uploader("Nạp PDF & Excel mới", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'], key=st.session_state.up_key)
    
    if up_files and st.button("🚀 BẮT ĐẦU NẠP & XỬ LÝ"):
        pdfs = [f for f in up_files if f.name.lower().endswith('.pdf')]
        exls = [f for f in up_files if f.name.lower().endswith(('.xls', '.xlsx'))]
        
        for f_p in pdfs:
            ma_p = set(re.findall(r'\d{3,}', f_p.name))
            f_e = None
            for ex in exls:
                ma_e = set(re.findall(r'\d{3,}', ex.name))
                common = ma_p.intersection(ma_e)
                if common:
                    f_e = ex; ma = max(common, key=len); break
            
            if f_p and f_e:
                with st.spinner(f"Đang xử lý mã {ma}..."):
                    with open("tmp.pdf", "wb") as t: t.write(f_p.getbuffer())
                    d, ex_img = extract_pdf("tmp.pdf"), excel_to_img(f_e)
                    if d and ex_img:
                        vec = get_vector(d['img']).tolist()
                        try:
                            supabase.storage.from_(BUCKET).upload(f"{ma}_t.webp", d['img'], {"x-upsert": "true"})
                            supabase.storage.from_(BUCKET).upload(f"{ma}_e.webp", ex_img, {"x-upsert": "true"})
                            u_t, u_e = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_t.webp"), supabase.storage.from_(BUCKET).get_public_url(f"{ma}_e.webp")
                            supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": u_t, "excel_img_url": u_e}, on_conflict="file_name").execute()
                            st.toast(f"✅ Xong mã {ma}")
                        except Exception as e: st.error(f"Lỗi: {e}")
                if os.path.exists("tmp.pdf"): os.remove("tmp.pdf")
        st.session_state.up_key += 1; st.rerun()

# ================= MAIN UI =================
st.title("👔 AI FASHION PRO - SO SÁNH THÔNG MINH")
test_pdf = st.file_uploader("1. Tải PDF Test", type="pdf")
target = st.session_state.target

if test_pdf:
    with open("test.pdf", "wb") as f: f.write(test_pdf.getbuffer())
    data_test = extract_pdf("test.pdf")
    if data_test:
        if st.button("🤖 AI: TỰ ĐỘNG TÌM MÃ GIỐNG NHẤT"):
            test_vec = get_vector(data_test['img'])
            best_sim, best_s = -1, None
            for s in samples:
                if s.get('vector'):
                    sim = cosine_similarity([test_vec], [np.array(s['vector'])])
                    if sim > best_sim: best_sim, best_s = sim, s
            st.session_state.target = best_s; st.rerun()

        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1: st.image(data_test['img'], caption="🖼️ ẢNH TEST", use_container_width=True)
        with c2:
            if target:
                st.image(target['img_url'], caption=f"📁 KHO ({target['file_name']})", use_container_width=True)
                if target.get('excel_img_url'): st.divider(); st.image(target['excel_img_url'], caption="📊 ĐỊNH MỨC KHO", use_container_width=True)
        with c3:
            if target:
                st.subheader(f"📊 Đối chiếu: {target['file_name']}")
                rows = []
                for k_t, v_t in data_test['spec'].items():
                    best_m, high_r = None, 0
                    for k_db in target['spec_json'].keys():
                        r = SequenceMatcher(None, k_t, k_db).ratio()
                        if r > high_r: high_r, best_m = r, k_db
                    v_db = target['spec_json'].get(best_m, 0) if high_r > 0.6 else "N/A"
                    diff = round(v_t - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"
                    rows.append({"Thông số PDF": k_t, "Test": v_t, "Kho": v_db, "Chênh lệch": diff, "Trạng thái": "✅ OK" if diff == 0 else "❌ SAI"})
                df = pd.DataFrame(rows)
                if not df.empty:
                    st.dataframe(df.style.map(lambda x: 'background-color: #ffcccc' if x == "❌ SAI" else ('background-color: #ccffcc' if x == "✅ OK" else ''), subset=['Trạng thái']), use_container_width=True, height=550)
                output = io.BytesIO()
                df.to_excel(output, index=False); st.download_button("📥 XUẤT EXCEL", output.getvalue(), f"Result_{target['file_name']}.xlsx")
