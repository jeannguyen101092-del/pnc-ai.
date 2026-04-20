import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
import requests
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE (SIẾT CHẶT TÌM KIẾM) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        img = img.crop((w*0.15, h*0.1, w*0.85, h*0.55)) 
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.0).convert('RGB')

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
        if not t or any(x in t for x in ["wash", "color", "label", "page", "tol", "+", "-"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if 0.1 <= val < 150 else 0
        return 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SCRAPER (FULL PAGE & NO TOL) =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_pil.save(buf, format="WEBP", quality=70)
        img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)
                size_cols = []
                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    texts = [str(t).strip() for t in sorted_group['text']]
                    valid_tokens = []
                    for t in texts:
                        t_clean = t.lower()
                        if t_clean in ["grade", "tol", "+tol", "-tol", "pom"]: continue
                        if re.match(r'^(xs|s|m|l|xl|xxl)$', t_clean) or re.match(r'^\d{1,3}$', t_clean):
                            valid_tokens.append(t)
                    if len(valid_tokens) >= 4:
                        for _, row in sorted_group.iterrows():
                            txt = row['text'].strip()
                            if txt in valid_tokens:
                                size_cols.append({"sz": txt.upper(), "x0": row['x0'] - 10, "x1": row['x1'] + 10})
                        break
                if not size_cols: continue

                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    line_txt = " ".join(sorted_group['text']).upper()
                    if any(x in line_txt for x in ["COVER", "IMAGE", "DATE", "CONSTRUCTION", "MEASUREMENT"]): continue
                    left_boundary = min([c['x0'] for c in size_cols])
                    left_part = sorted_group[sorted_group['x0'] < left_boundary]
                    pom_name = " ".join(left_part['text']).strip()
                    if len(pom_name) < 3: continue

                    for col in size_cols:
                        cell = sorted_group[(sorted_group['x0'] >= col['x0']) & (sorted_group['x1'] <= col['x1'])]
                        if not cell.empty:
                            raw = " ".join(cell['text'])
                            val = None
                            try:
                                if "/" in raw:
                                    parts = raw.split()
                                    total = 0
                                    for p in parts:
                                        if "/" in p:
                                            num, den = p.split("/")
                                            total += float(num) / float(den)
                                        else: total += float(p)
                                    val = total
                                else: val = float(raw)
                            except: continue
                            if val is not None:
                                if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        count = res_count.count or 0
    except: count = 0
    st.metric("Models in Repo", f"{count} SKUs")
    
    with st.expander("⚙️ Bảo trì & Nâng cấp AI"):
        if st.button("🚀 BẮT ĐẦU NÂNG CẤP", use_container_width=True):
            items = supabase.table("ai_data").select("id, image_url").execute().data
            if items:
                p_bar = st.progress(0)
                for i, it in enumerate(items):
                    try:
                        r = requests.get(it['image_url'], timeout=10)
                        if r.status_code == 200:
                            v = get_vector(r.content)
                            if v: supabase.table("ai_data").update({"vector": v}).eq("id", it['id']).execute()
                        p_bar.progress((i+1)/len(items))
                    except: continue
                st.success("Xong!"); st.rerun()

    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        st.info("Đang đồng bộ...")
        for f in new_files:
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                fname = f"{uuid.uuid4()[:8]}_{f.name.replace(' ', '_')}.webp"
                supabase.storage.from_(BUCKET).upload(f"sketches/{fname}", data['img'])
                img_url = supabase.storage.from_(BUCKET).get_public_url(f"sketches/{fname}")
                supabase.table("ai_data").insert({"file_name": f.name, "image_url": img_url, "vector": get_vector(data['img']), "specs": data['all_specs']}).execute()
        st.success("Đã đồng bộ!"); st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["Audit Mode", "Version Control"], horizontal=True)

if mode == "Audit Mode":
    st.subheader("🔍 Tìm kiếm mẫu tương đồng")
    up_target = st.file_uploader("Upload Target PDF", type=['pdf'], key=f"t_{st.session_state['up_key']}")
    if up_target:
        st.info("Đang xử lý...")

elif mode == "Version Control":
    st.subheader("🔄 So sánh 2 file PDF (ALL SIZE)")
    if st.button("🗑️ Xoá file đã upload", use_container_width=True):
        st.session_state['up_key'] += 1; st.session_state['ver_results'] = None; st.rerun()
    
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Bản cũ (A):", type="pdf", key=f"v1_{st.session_state['up_key']}")
    f2 = c2.file_uploader("Bản mới (B):", type="pdf", key=f"v2_{st.session_state['up_key']}")

    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh toàn diện", use_container_width=True):
            with st.spinner("Đang quét..."):
                d1, d2 = extract_full_data(f1.getvalue()), extract_full_data(f2.getvalue())
                if d1 and d2: st.session_state['ver_results'] = {"d1": d1, "d2": d2, "f1_name": f1.name, "f2_name": f2.name}

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider()
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())))
        version_dfs, ver_sheets = [], []

        def color_status(val):
            if val == "❌ Lệch": return 'background-color: #ffcccc; color: #990000; font-weight: bold;'
            if val == "✅ Khớp": return 'background-color: #ccffcc; color: #006600;'
            return 'background-color: #fff3cd; color: #856404;'

        for sz in all_sz:
            with st.expander(f"📊 SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = []
                for p in poms:
                    v1, v2 = s1.get(p), s2.get(p)
                    if v1 is not None and v2 is not None:
                        diff = round(v2 - v1, 3)
                        st_val = "✅ Khớp" if abs(diff) < 0.001 else "❌ Lệch"
                        dt = f"{diff:+.3f}"
                    else: dt, st_val = "N/A", "⚠️ Thiếu"
                    rows.append({"POM": p, "Bản A": v1, "Bản B": v2, "Chênh lệch": dt, "Kết quả": st_val})
                df_sz = pd.DataFrame(rows)
                try: st.dataframe(df_sz.style.map(color_status, subset=['Kết quả']), use_container_width=True)
                except: st.dataframe(df_sz.style.applymap(color_status, subset=['Kết quả']), use_container_width=True)
                version_dfs.append(df_sz); ver_sheets.append(f"Size_{sz}")

        if version_dfs:
            if st.download_button("📥 Xuất Excel So Sánh", to_excel(version_dfs, ver_sheets), "Comparison.xlsx", use_container_width=True):
                st.success("Đã sẵn sàng tải xuống!")
