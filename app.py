import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state: st.session_state['reset_key'] = 0
if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None

# ================= 2. AI CORE ENGINE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes): return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds|tol|grade)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

def translate_insight(text):
    mapping = {"WASH": "HD Giặt", "FABRIC": "Vải", "STITCH": "May", "LABEL": "Nhãn", "COLOR": "Màu", "POCKET": "Túi", "WAIST": "Lưng"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

# ================= 3. PDF EXTRACTION (SKETCH ISOLATION) =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary, category = {}, None, "", "UNKNOWN"
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        
        # Chiến thuật trích xuất Sketch: Chỉ lấy vùng có nét vẽ, bỏ qua khung bảng
        paths = page.get_drawings()
        if paths:
            bboxes = [p["rect"] for p in paths if 50 < p["rect"].width < page.rect.width * 0.8]
            if bboxes:
                x0, y0 = min([b.x0 for b in bboxes]), min([b.y0 for b in bboxes])
                x1, y1 = max([b.x1 for b in bboxes]), max([b.y1 for b in bboxes])
                crop = fitz.Rect(max(0, x0-20), max(0, y0-20), min(page.rect.width, x1+20), min(page.rect.height, y1+20))
                pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=crop)
            else: pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        else: pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        
        img_t = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_t.save(buf, format="WEBP", quality=80); img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for p in pdf.pages:
                txt = p.extract_text() or ""
                full_txt += txt + "\n"
                for tb in p.extract_tables():
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    d_col = df.apply(lambda x: x.astype(str).str.len().mean()).idxmax()
                    for c_idx in range(len(df.columns)):
                        if c_idx == d_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, c_idx].head(15)]) > 0:
                            s_n = str(df.iloc[0, c_idx]).strip().replace('\n', ' ')
                            specs = {str(df.iloc[d, d_col]).strip(): parse_val(df.iloc[d, c_idx]) for d in range(len(df)) if len(str(df.iloc[d, d_col])) > 3}
                            if specs:
                                if s_n not in all_specs: all_specs[s_n] = {}
                                all_specs[s_n].update(specs)
            
            # --- AI CATEGORY DETECTIVE ---
            txt_upper = full_txt.upper()
            if any(x in txt_upper for x in ["PANT", "QUAN", "TROUSER", "SHORT", "JEAN"]): category = "BOTTOM"
            elif any(x in txt_upper for x in ["SHIRT", "AO", "JACKET", "TOP", "TEE", "POLO"]): category = "TOP"
            elif "SKIRT" in txt_upper or "VAY" in txt_upper: category = "SKIRT"

            vi_summary = translate_insight(full_txt)

        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary, "category": category}
    except: return None

# ================= 4. UI SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Repository Models", f"{count} SKUs")
    st.write(f"💾 **Storage:** {count * 0.07:.1f}MB / 1024MB")
    st.progress(min((count / 10000), 1.0))
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE", use_container_width=True):
        with st.spinner("AI Categorizing & Saving..."):
            for f in new_files:
                fb = f.read(); h = get_file_hash(fb)
                data = extract_full_techpack(fb)
                if data and data.get('img') and data.get('all_specs'):
                    path = f"lib_{h}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], 
                        "category": data['category'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
        st.rerun()
    if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI (HYBRID SMART SEARCH) =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode:", ["🔍 AI Search (Audit)", "🔄 Version Control"], horizontal=True)

if mode == "🔍 AI Search (Audit)":
    file_audit = st.file_uploader("Upload PDF to find match:", type="pdf")
    if file_audit:
        target = extract_full_techpack(file_audit.read())
        if target:
            with st.expander("📝 AI TECHNICAL INSIGHT (VIETNAMESE)", expanded=True):
                st.write(target['summary_vi'] if target['summary_vi'] else "No notes found.")
            
            res = supabase.table("ai_data").select("*").execute()
            df_db = pd.DataFrame(res.data)
            
            # --- SUPER SENSITIVE HYBRID FILTER ---
            t_cat = target['category']
            t_name = file_audit.name.upper()

            def final_scoring(row):
                # 1. Nếu khác loại hàng (Áo vs Quần) -> Điểm 0 (Loại bỏ)
                if t_cat != "UNKNOWN" and row['category'] != "UNKNOWN" and t_cat != row['category']:
                    return 0.0
                
                # 2. Tính Bonus dựa trên từ khóa chi tiết (Cargo, Elastic...)
                bonus = 1.0
                row_name = str(row['file_name']).upper()
                for kw in ["CARGO", "THUN", "ELASTIC", "WAIST", "POCKET"]:
                    if (kw in t_name) == (kw in row_name): bonus += 0.2
                return bonus

            df_db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in df_db['vector']])).flatten()
            df_db['final_score'] = df_db['sim'] * df_db.apply(final_scoring, axis=1)
            
            top_3 = df_db[df_db['final_score'] > 0].sort_values('final_score', ascending=False).head(3)
            
            st.subheader("🎯 Intelligent Matches (Filtered by Category)")
            if top_3.empty:
                st.warning(f"No match found for category: {t_cat}")
            else:
                cols = st.columns(4)
                with cols[0]: st.image(target['img'], caption=f"TARGET ({t_cat})", use_container_width=True)
                for i, (idx, row) in enumerate(top_3.iterrows()):
                    with cols[i+1]:
                        st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}")
                        if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}"): st.session_state['sel_audit'] = row.to_dict()

            selected = st.session_state.get('sel_audit')
            if selected:
                all_ex = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = selected['spec_json'].get(sz, {})
                        rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        for r in rows:
                            diff = r['Target'] - r['Ref']
                            r['Diff'] = f"{diff:+.3f}"
                            all_ex.append({"Size": sz, **r, "Diff_Val": diff})
                        st.table(pd.DataFrame(rows))
                
                buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                st.download_button("📥 DOWNLOAD REPORT", buf.getvalue(), f"Audit_{selected['file_name']}.xlsx", use_container_width=True)
else:
    st.info("Version Control mode active. Use sidebar to manage rounds.")
