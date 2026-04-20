import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
# Thay URL và KEY của bạn vào đây
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
    all_specs, all_imgs = {}, []
    try:
        # Lấy ảnh minh họa tất cả các trang
        doc = fitz.open(stream=file_content, filetype="pdf")
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            all_imgs.append(pix.tobytes("png"))
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
                    valid_tokens = [t.strip() for t in sorted_group['text'] if re.match(r'^(xs|s|m|l|xl|xxl)$', t.strip().lower()) or re.match(r'^\d{1,3}$', t.strip())]
                    if len(valid_tokens) >= 4:
                        for _, row in sorted_group.iterrows():
                            if row['text'].strip() in valid_tokens:
                                size_cols.append({"sz": row['text'].strip().upper(), "x0": row['x0'] - 10, "x1": row['x1'] + 10})
                        break

                if not size_cols: continue

                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    line_txt = " ".join(sorted_group['text']).upper()
                    if any(x in line_txt for x in ["COVER", "IMAGE", "DATE", "CONSTRUCTION"]): continue
                    
                    left_boundary = min([c['x0'] for c in size_cols])
                    pom_name = " ".join(sorted_group[sorted_group['x0'] < left_boundary]['text']).strip()
                    if len(pom_name) < 3 or any(k in pom_name.lower() for k in ["poly", "bag", "sticker"]): continue

                    for col in size_cols:
                        cell = sorted_group[(sorted_group['x0'] >= col['x0']) & (sorted_group['x1'] <= col['x1'])]
                        if not cell.empty:
                            val = parse_val(" ".join(cell['text']))
                            if val > 0:
                                if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "imgs": all_imgs}
    except: return None

# ================= 4. SIDEBAR & MENU =================
with st.sidebar:
    st.markdown("<h1 style='color:#1E88E5;'>👔 PPJ AI Auditor</h1>", unsafe_allow_html=True)
    menu = ["🔍 Tìm kiếm tương đồng", "🔄 Version Control"]
    mode = st.selectbox("Chức năng chính:", menu)

# ================= 5. SIMILARITY SEARCH (PHẦN SỬA) =================
if mode == "🔍 Tìm kiếm tương đồng":
    st.subheader("🔍 Tìm kiếm thiết kế tương đồng (AI Match)")
    up_file = st.file_uploader("Tải lên bản vẽ (PDF/Ảnh):", type=["pdf", "png", "jpg", "jpeg"])
    
    if up_file:
        file_bytes = up_file.getvalue()
        if up_file.type == "application/pdf":
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                search_img = doc.load_page(0).get_pixmap().tobytes("png")
        else: search_img = file_bytes
        
        q_vec = get_vector(search_img)
        if q_vec:
            with st.spinner("Đang đối soát AI..."):
                res = supabase.table("fashion_audits").select("filename, img_url, vector_ai").execute()
                matches = []
                for item in res.data:
                    if item['vector_ai']:
                        score = cosine_similarity([q_vec], [item['vector_ai']])[0][0]
                        if score > 0.6: matches.append({"name": item['filename'], "url": item['img_url'], "score": score})
                
                matches = sorted(matches, key=lambda x: x['score'], reverse=True)
                if matches:
                    cols = st.columns(4)
                    for i, m in enumerate(matches[:12]):
                        with cols[i % 4]:
                            st.image(m['url'], use_container_width=True)
                            st.write(f"**Giống: {int(m['score']*100)}%**")
                            st.caption(m['name'][:20])
                else: st.warning("Không tìm thấy mẫu tương đồng.")

# ================= 6. VERSION CONTROL =================
elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh dữ liệu giữa 2 File PDF")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("File A (Gốc):", type="pdf", key="v1")
    f2 = c2.file_uploader("File B (Mới):", type="pdf", key="v2")

    if f1 and f2 and st.button("⚡ Bắt đầu so sánh toàn diện"):
        with st.spinner("Đang quét toàn bộ các trang..."):
            st.session_state['ver_results'] = {"d1": extract_full_data(f1.getvalue()), "d2": extract_full_data(f2.getvalue()), "n1": f1.name, "n2": f2.name}

    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        with st.expander("🖼️ Xem ảnh các trang đã quét"):
            t1, t2 = st.tabs(["File A", "File B"])
            t1.image(vr['d1']['imgs'], caption=[f"A-P{i+1}" for i in range(len(vr['d1']['imgs']))], width=200)
            t2.image(vr['d2']['imgs'], caption=[f"B-P{i+1}" for i in range(len(vr['d2']['imgs']))], width=200)

        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())))
        dfs, sheets = [], []
        for sz in all_sz:
            with st.expander(f"📏 Size: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                poms = sorted(list(set(s1.keys()) | set(s2.keys())))
                rows = [{"POM": p, "A": s1.get(p,0), "B": s2.get(p,0), "Diff": f"{s2.get(p,0)-s1.get(p,0):+.2f}", "Status": "✅" if s1.get(p,0)==s2.get(p,0) else "⚠️"} for p in poms]
                df = pd.DataFrame(rows)
                st.table(df)
                dfs.append(df); sheets.append(sz)
        st.download_button("📥 Xuất Excel", to_excel(dfs, sheets), "SoSanh.xlsx")
