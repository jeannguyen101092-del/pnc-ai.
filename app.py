import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
import requests
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'up_key' not in st.session_state: st.session_state['up_key'] = 0
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None

# ================= 2. AI CORE & SCRAPER =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        img = img.crop((w*0.20, h*0.12, w*0.80, h*0.50)) 
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.5).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            norm = np.linalg.norm(vec)
            return (vec / norm).astype(float).tolist() if norm > 0 else vec.tolist()
    except: return None

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        trash = ["poly", "bag", "twill", "paper", "label", "button", "frisbee", "seaman"]
        if not t or any(x in t for x in trash): return 0
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if 0.2 <= val < 150 else 0
        return 0
    except: return 0

def extract_full_data(file_content):
    all_specs, img_bytes = {}, None
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        doc.close()
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)
                size_cols = []
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text']).lower()
                    if any(x in line_txt for x in ["size", "spec", "adopted"]):
                        for _, row in group.iterrows():
                            txt = row['text'].strip().lower()
                            if re.match(SIZE_PATTERN, txt) and row['x0'] > 250:
                                size_cols.append({"sz": txt.upper(), "x0": row['x0']-10, "x1": row['x1']+10})
                        if size_cols: break
                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    pom_name = " ".join(sorted_group[sorted_group['x1'] < 350]['text']).strip()
                    if 3 < len(pom_name) < 65 and not any(x in pom_name.upper() for x in ["STYLE", "DATE", "FABRIC", "PAGE"]):
                        for col in size_cols:
                            cell = sorted_group[(sorted_group['x0'] >= col['x0']) & (sorted_group['x1'] <= col['x1'])]
                            if not cell.empty:
                                val = parse_val(" ".join(cell['text']))
                                if val > 0:
                                    if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                    all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    try:
        count = (supabase.table("ai_data").select("id", count="exact").execute()).count or 0
    except: count = 0
    st.metric("Models in Repo", f"{count} SKUs")
    
    with st.expander("⚙️ Nâng cấp AI"):
        if st.button("🚀 BẮT ĐẦU", use_container_width=True):
            items = (supabase.table("ai_data").select("id, image_url").execute()).data
            if items:
                p_bar = st.progress(0)
                for i, item in enumerate(items):
                    try:
                        r = requests.get(item['image_url'], timeout=10)
                        if r.status_code == 200:
                            v = get_vector(r.content)
                            if v: supabase.table("ai_data").update({"vector": v}).eq("id", item['id']).execute()
                        p_bar.progress((i+1)/len(items))
                    except: continue
                st.success("Xong!"); st.rerun()

    st.divider()
    # Key ở đây là duy nhất cho Sidebar
    up_new = st.file_uploader("Nạp kho mẫu mới", accept_multiple_files=True, key=f"side_up_{st.session_state['up_key']}")
    if up_new and st.button("🚀 ĐẨY VÀO KHO", use_container_width=True):
        st.info("Đang xử lý...")
        # Code đẩy dữ liệu của bạn ở đây...

# ================= 4. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["Audit Mode", "Version Control"], horizontal=True)

if mode == "Audit Mode":
    st.subheader("🔍 Tìm kiếm tương đồng")
    # Key ở đây khác hoàn toàn với Sidebar
    target = st.file_uploader("Upload bản vẽ cần tìm", type=['pdf'], key=f"audit_{st.session_state['up_key']}")
    if target: st.write("Đang tìm mẫu giống với:", target.name)

elif mode == "Version Control":
    st.subheader("🔄 So sánh 2 file PDF")
    if st.button("🗑️ Làm mới", use_container_width=True):
        st.session_state['up_key'] += 1; st.session_state['ver_results'] = None; st.rerun()
    
    c1, c2 = st.columns(2)
    # Các Key ở đây cũng được đặt tên riêng biệt (v1_, v2_)
    f1 = c1.file_uploader("Bản cũ (A)", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B)", type="pdf", key=f"v2_{st.session_state['up_key']}")

    if f1 and f2:
        if st.button("⚡ So sánh ngay", use_container_width=True):
            with st.spinner("Đang quét..."):
                d1, d2 = extract_full_data(f1.getvalue()), extract_full_data(f2.getvalue())
                if d1 and d2: st.session_state['ver_results'] = {"d1": d1, "d2": d2, "n1": f1.name, "n2": f2.name}
                else: st.error("Lỗi đọc dữ liệu")

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider()
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())))
        version_dfs, ver_sheets = [], []

        for sz in all_sz:
            with st.expander(f"📊 SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = []
                for p in poms:
                    v1, v2 = s1.get(p), s2.get(p)
                    if v1 and v2:
                        diff = round(v2 - v1, 3)
                        status = "✅ Khớp" if abs(diff) < 0.001 else "❌ Lệch"
                        dt = f"{diff:+.3f}"
                    else: dt, status = "N/A", "⚠️ Thiếu"
                    rows.append({"POM": p, "Bản A": v1, "Bản B": v2, "Lệch": dt, "Kết quả": status})
                df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)
                version_dfs.append(df); ver_sheets.append(f"Size_{sz}")
        
        if version_dfs:
            st.download_button("📥 Tải Excel", to_excel(version_dfs, ver_sheets), "Compare.xlsx", use_container_width=True)
