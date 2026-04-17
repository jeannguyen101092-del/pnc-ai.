import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
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
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE =================
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

# ================= 3. SCRAPER =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_pil.save(buf, format="WEBP", quality=70); img_bytes = buf.getvalue(); doc.close()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)
                
                size_cols = []
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text']).lower()
                    if "size" in line_txt or "adopted" in line_txt:
                        for _, row in group.iterrows():
                            txt = row['text'].strip().lower()
                            if re.match(SIZE_PATTERN, txt) and txt not in ["tol", "um", "(+)", "(-)"]:
                                size_cols.append({"sz": txt.upper(), "x0": row['x0']-5, "x1": row['x1']+5})
                        if size_cols: break

                for y, group in df_w.groupby('y_grid'):
                    sorted_group = group.sort_values('x0')
                    line_txt = " ".join(sorted_group['text']).upper()
                    if any(x in line_txt for x in ["COVER", "IMAGE", "DATE", "CONSTRUCTION"]): continue
                    
                    pom_name = re.sub(r'[\d./\s]+$', '', " ".join(sorted_group[sorted_group['x1'] < 350]['text'])).strip()
                    if len(pom_name) > 3:
                        for col in size_cols:
                            cell = sorted_group[(sorted_group['x0'] >= col['x0']) & (sorted_group['x1'] <= col['x1'])]
                            if not cell.empty:
                                val = parse_val(" ".join(cell['text']))
                                if val > 0:
                                    if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                    all_specs[col['sz']][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Models in Repo", f"{count} SKUs")
    
    storage_mb = count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    st.divider()
    
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        prog_bar = st.progress(0)
        for i, f in enumerate(new_files):
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "vector": get_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
            prog_bar.progress((i + 1) / len(new_files))
        st.success("✅ Đồng bộ hoàn tất!")
        st.session_state['up_key'] += 1
        st.rerun()

# ================= 5. MAIN UI (PHẦN SỬA ĐỔI THÔNG MINH NHẤT) =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue())
        if target and target['img']:
            res = supabase.table("ai_data").select("id, vector, file_name, image_url, spec_json").execute()
            
            if res.data:
                # --- LOGIC TÌM KIẾM THÔNG MINH ---
                t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                db_vectors = np.array([r['vector'] for r in res.data])
                db_names = [r['file_name'] for r in res.data]
                
                # 1. Tính độ tương đồng hình ảnh
                sim_scores = cosine_similarity(t_vec, db_vectors)[0]
                
                # 2. Hybrid Scoring (Ảnh + Tên file)
                matches = []
                for i, score in enumerate(sim_scores):
                    name_sim = len(get_close_matches(f_audit.name.upper(), [db_names[i].upper()], cutoff=0.5))
                    final_score = (score * 0.8) + (0.2 if name_sim > 0 else 0)
                    matches.append({**res.data[i], "score": final_score})
                
                # 3. Lọc & Sắp xếp
                matches = sorted([m for m in matches if m['score'] > 0.65], key=lambda x: x['score'], reverse=True)[:5]

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("🎯 Target Sketch")
                    st.image(target['img'], use_container_width=True)
                
                with col2:
                    st.subheader("📂 AI Suggested Matches")
                    if matches:
                        for m in matches:
                            with st.expander(f"📌 {m['file_name']} (Khớp: {m['score']:.1%})"):
                                st.image(m['image_url'])
                                if st.button("So sánh thông số", key=m['id']):
                                    st.session_state['sel_audit'] = m
                    else:
                        st.error("Không tìm thấy mẫu tương đồng phù hợp.")

# ================= 6. BÁO CÁO AUDIT (SPEC COMPARISON) =================
if st.session_state['sel_audit']:
    st.divider()
    m = st.session_state['sel_audit']
    st.header(f"📊 Audit Report: {m['file_name']}")
    
    db_spec, tg_spec = m['spec_json'], target['all_specs']
    common_sizes = sorted(list(set(db_spec.keys()) & set(tg_spec.keys())))
    
    if common_sizes:
        sz = st.selectbox("Chọn Size kiểm tra:", common_sizes)
        df1 = pd.DataFrame(db_spec[sz].items(), columns=['POM', 'Standard'])
        df2 = pd.DataFrame(tg_spec[sz].items(), columns=['POM', 'Actual'])
        
        report = pd.merge(df1, df2, on='POM', how='inner')
        report['Diff'] = (report['Actual'] - report['Standard']).round(3)
        
        def style_diff(val):
            return 'background-color: #ffcccc; color: red' if abs(val) > 0.25 else 'background-color: #ccffcc; color: green'

        st.dataframe(report.style.applymap(style_diff, subset=['Diff']), use_container_width=True)
        
        btn_xlsx = to_excel([report], [f"Audit_{sz}"])
        st.download_button("📥 Xuất File Excel", btn_xlsx, f"Audit_{m['file_name']}.xlsx")
    else:
        st.warning("⚠️ Không có dữ liệu Size tương đồng để so sánh.")
