import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, random
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG (Thay URL và KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("❌ Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V11.49", page_icon="👔")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.features.children()) + [torch.nn.AdaptiveAvgPool2d(1)])).eval()

ai_brain = load_ai()

# ================= HÀM HỖ TRỢ AI =================
def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        return ai_brain(tf(img).unsqueeze(0)).flatten().numpy()

def parse_val(t):
    try:
        t_str = str(t).strip().replace('-', ' ')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).dropna(how='all', axis=0).fillna("")
        fig, ax = plt.subplots(figsize=(20, len(df.head(60)) * 0.6 + 2)) 
        ax.axis('off')
        table = ax.table(cellText=df.head(60).values, colLabels=df.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(14); table.scale(1.2, 2.8) 
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200); plt.close(fig)
        return buf.getvalue()
    except: return None

def get_data(pdf_path):
    try:
        specs, text, base_size = {}, "", "8"
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: 
                    text += t
                    m = re.search(r'Base Size\s*[:\s]\s*(\w+[\-?]?)', t, re.I)
                    if m: base_size = m.group(1).upper()
            for p in pdf.pages:
                for tb in p.extract_tables():
                    if not tb or len(tb) < 2: continue
                    h_idx = -1
                    for i, row in enumerate(tb[:10]):
                        row_up = [str(x or "").strip().upper() for x in row]
                        if any(base_size in x for x in row_up):
                            h_idx, header = i, row_up; break
                    if h_idx == -1: continue
                    base_idx = -1
                    for idx, h_val in enumerate(header):
                        if base_size in h_val and len(h_val) < 6: base_idx = idx
                    if base_idx != -1:
                        for r in tb[h_idx + 1:]:
                            if not r or len(r) <= base_idx: continue
                            desc = " ".join([str(x or "") for x in r[:base_idx]]).strip().upper()
                            val = parse_val(r[base_idx])
                            if val > 0.1 and len(desc) > 5:
                                specs[re.sub(r'[^A-Z0-9\s/]', '', desc)[:150]] = round(float(val), 3)
        doc = fitz.open(pdf_path); img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png"); doc.close()
        return {"spec": specs, "img": img}
    except: return None

# ================= SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("*").execute()
        all_samples = db_res.data
    except: all_samples = []
    
    st.metric("Tổng mẫu trong kho", f"{len(all_samples)} mẫu")
    
    list_ma = [item['file_name'] for item in all_samples]
    selected_ma = st.selectbox("🎯 CHỌN MÃ HÀNG CỐ ĐỊNH", ["-- Chọn mã --"] + list_ma)
    if selected_ma != "-- Chọn mã --":
        st.session_state.target_sample = next(item for item in all_samples if item['file_name'] == selected_ma)
        st.session_state.match_score = 100.0
    
    if st.button("🎲 CHỌN MÃ NGẪU NHIÊN") and all_samples:
        st.session_state.target_sample = random.choice(all_samples)
        st.session_state.match_score = 0.0

    st.divider()
    files = st.file_uploader("Nạp PDF & Excel mới", accept_multiple_files=True, type=['pdf', 'xlsx'], key=f"up_{st.session_state.uploader_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        groups = {}
        for f in files:
            m = re.search(r'^\d+', f.name)
            if m:
                ma = m.group(); ext = os.path.splitext(f.name)[1].lower()
                if ma not in groups: groups[ma] = {}
                groups[ma][ext] = f
        for ma, parts in groups.items():
            f_p, f_e = parts.get('.pdf'), (parts.get('.xlsx') or parts.get('.xls'))
            if f_p and f_e:
                with st.spinner(f"Đang nạp mã: {ma}"):
                    with open("tmp.pdf", "wb") as t: t.write(f_p.getbuffer())
                    d, exl = get_data("tmp.pdf"), excel_to_img_bytes(f_e)
                    if d and exl:
                        vec = get_vector(d['img']).tolist()
                        try:
                            supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_t.webp", d['img'], {"content-type": "image/webp", "x-upsert": "true"})
                            supabase.storage.from_(BUCKET_NAME).upload(f"{ma}_e.webp", exl, {"content-type": "image/webp", "x-upsert": "true"})
                            url_t = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_t.webp")
                            url_e = supabase.storage.from_(BUCKET_NAME).get_public_url(f"{ma}_e.webp")
                            supabase.table("ai_data").upsert({"file_name": ma, "vector": vec, "spec_json": d['spec'], "img_url": url_t, "excel_img_url": url_e}, on_conflict="file_name").execute()
                        except: pass
        st.session_state.uploader_key += 1; st.rerun()

# ================= MAIN UI =================
st.title("👔 AI Fashion Pro - So Sánh Thông Số")
test_file = st.file_uploader("Tải PDF Test", type="pdf")
target = st.session_state.get('target_sample')

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    data_test = get_data("test.pdf")
    
    if data_test:
        # --- LOGIC TÌM MÃ GIỐNG NHẤT BẰNG AI ---
        if st.button("🤖 TỰ ĐỘNG TÌM MÃ TƯƠNG ĐỒNG NHẤT"):
            test_vec = get_vector(data_test['img'])
            best_sim, best_item = -1, None
            for item in all_samples:
                if item.get('vector'):
                    sim = cosine_similarity([test_vec], [np.array(item['vector'])])[0][0]
                    if sim > best_sim:
                        best_sim, best_item = sim, item
            if best_item:
                st.session_state.target_sample = best_item
                st.session_state.match_score = round(best_sim * 100, 1)
                st.rerun()

        # HIỂN THỊ 3 CỘT (Cấu trúc cũ)
        col_test, col_target, col_info = st.columns([1, 1, 1.5])
        with col_test:
            st.image(data_test['img'], caption="🖼️ 1. ẢNH ĐANG TEST", use_container_width=True)
        with col_target:
            if target:
                score = st.session_state.get('match_score', 0)
                st.image(target['img_url'], caption=f"📁 2. ẢNH KHO ({target['file_name']}) - GIỐNG {score}%", use_container_width=True)
                if target.get('excel_img_url'):
                    st.divider(); st.image(target['excel_img_url'], caption="📊 3. ĐỊNH MỨC EXCEL KHO", use_container_width=True)
            else: st.warning("👈 Chọn mã ở Sidebar hoặc bấm nút Tìm mã tương đồng")
        
        with col_info:
            if target:
                st.subheader(f"📊 Đối chiếu: {target['file_name']}")
                rows = []
                test_specs, db_specs = data_test['spec'], target['spec_json']
                for k_test, v_test in test_specs.items():
                    best_match, highest_ratio = None, 0
                    for k_db in db_specs.keys():
                        ratio = SequenceMatcher(None, k_test, k_db).ratio()
                        if ratio > highest_ratio: highest_ratio, best_match = ratio, k_db
                    v_db = db_specs.get(best_match, 0) if highest_ratio > 0.6 else "N/A"
                    diff = round(v_test - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"
                    status = "✅ OK" if diff == 0 else ("❌ SAI" if diff != "N/A" else "❓ MỚI")
                    rows.append({"Thông số PDF Test": k_test, "Số đo Test": v_test, "Số đo Kho": v_db, "Chênh lệch": diff, "Trạng thái": status})
                df_compare = pd.DataFrame(rows)
                st.dataframe(df_compare.style.map(lambda x: 'background-color: #ffcccc' if x == "❌ SAI" else ('background-color: #ccffcc' if x == "✅ OK" else ''), subset=['Trạng thái']), use_container_width=True, height=550)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df_compare.to_excel(writer, index=False)
                st.download_button("📥 XUẤT EXCEL", output.getvalue(), f"SoSanh_{target['file_name']}.xlsx")
