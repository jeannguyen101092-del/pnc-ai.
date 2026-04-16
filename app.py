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
if 'sel' not in st.session_state: st.session_state['sel'] = None

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
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

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

# ================= 3. PDF EXTRACTION (HIGH-DPI & WEB-P) =================
def extract_pdf_full_logic(file_content):
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        # High resolution for Zoom (DPI 2.5)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_temp = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_temp.save(buf, format="WEBP", quality=85)
        img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages:
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    char_counts = df.apply(lambda x: x.astype(str).str.len().mean())
                    desc_col = char_counts.idxmax()
                    for col_idx in range(len(df.columns)):
                        if col_idx == desc_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, col_idx].head(10)]) > 0:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ')
                            temp_data = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                val = parse_val(df.iloc[d_idx, col_idx])
                                if val > 0 and len(pom) > 2: temp_data[pom] = val
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SAFE CLEANUP LOGIC =================
def safe_deep_clean():
    try:
        files = supabase.storage.from_(BUCKET).list(options={"limit": 5000})
        db_res = supabase.table("ai_data").select("id").execute()
        db_ids = [r['id'] for r in db_res.data]
        deleted = 0
        for f in files:
            name = f['name']
            f_id = name.replace("lib_", "").replace(".webp", "").replace(".png", "")
            if f_id not in db_ids or f['metadata'].get('size', 0) < 1500:
                try:
                    supabase.storage.from_(BUCKET).remove([name])
                    supabase.table("ai_data").delete().eq("id", f_id).execute()
                    deleted += 1
                except: continue
        return deleted
    except: return 0

# ================= 5. SIDEBAR (MANAGEMENT) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold; margin-bottom: 0;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666;'>AI Technical Auditor Pro</p>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("📂 Repository Status")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Models", f"{count} SKUs")
    
    # --- STORAGE DISPLAY (DUNG LƯỢNG) ---
    used_mb = (count * 0.07) # Approx 70KB per WebP
    st.write(f"💾 **Cloud Storage:** {used_mb:.1f}MB / 1024MB")
    st.progress(min((used_mb / 1024), 1.0))
    
    if st.button("🧹 CLEAN ERRORS & TRASH", use_container_width=True):
        num = safe_deep_clean()
        st.success(f"Cleaned {num} items!")
        time.sleep(1); st.rerun()

    st.divider()
    st.subheader("📥 Data Ingestion")
    new_files = st.file_uploader("Upload Tech-Packs (PDF)", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    c1, c2 = st.columns(2)
    with c1:
        if new_files and st.button("SYNCHRONIZE", use_container_width=True):
            with st.spinner("AI Processing..."):
                new_count = 0
                for f in new_files:
                    fb = f.read(); h = get_file_hash(fb)
                    check = supabase.table("ai_data").select("id").eq("id", h).execute()
                    if check.data:
                        st.sidebar.warning(f"⏩ {f.name} exists.")
                        continue
                    data = extract_pdf_full_logic(fb)
                    if data and data.get('img') and data.get('all_specs') and len(data['all_specs']) > 0:
                        path = f"lib_{h}.webp"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                        supabase.table("ai_data").upsert({
                            "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                            "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                        }).execute()
                        new_count += 1
                if new_count > 0: st.sidebar.success(f"Added {new_count} SKUs!")
            time.sleep(1); st.rerun()
    with c2:
        if st.button("CLEAR LIST", use_container_width=True):
            st.session_state['reset_key'] += 1; st.rerun()

# ================= 6. MAIN AUDIT UI =================
h_col1, h_col2 = st.columns([1, 5])
with h_col1:
    st.markdown("<h1 style='color: #1E3A8A; margin:0;'>PPJ</h1>", unsafe_allow_html=True)
with h_col2:
    st.title("SMART AUDITOR PRO")
    st.markdown("*Intelligent Technical Verification System*")

st.markdown("---")
file_audit = st.file_uploader("📤 Drag & Drop Target Tech-Pack for Auditing", type="pdf")

if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_full_logic(a_bytes)
    
    if target and target.get("img"):
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # --- MANUAL SEARCH ---
            st.subheader("🔍 Manual SKU Search")
            search_q = st.text_input("Search by SKU ID or Filename:", placeholder="Enter code...")
            if search_q:
                df_db = df_db[df_db['file_name'].str.contains(search_q, case=False, na=False)]
            
            if st.button("🚀 AUTO SEARCH (AI MODE)", use_container_width=True):
                st.session_state['sel'] = None

            # --- SENSITIVE FILTER LOGIC ---
            t_name = file_audit.name.upper()
            KEYWORDS = {"CARGO": ["CARGO", "HOP"], "WAIST": ["ELASTIC", "THUN"], "TYPE": ["SKIRT", "VAY", "PANT", "QUAN", "SHORT"]}
            def get_w(row_name):
                row_name = str(row_name).upper(); w = 1.0
                for kw in KEYWORDS["TYPE"]:
                    if (kw in t_name) == (kw in row_name): w += 0.5
                for kw in KEYWORDS["CARGO"] + KEYWORDS["WAIST"]:
                    if (kw in t_name) and (kw in row_name): w += 0.3
                return w

            # AI Calculation
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            df_db['score'] = cosine_similarity(target_vec, db_vecs).flatten() * df_db['file_name'].apply(get_w)
            top_3 = df_db.sort_values('score', ascending=False).head(3)
            
            if st.session_state['sel'] is None:
                st.subheader("🎯 AI Matching Results (Click to Zoom)")
                c_ai = st.columns(4)
                with c_ai[0]: st.image(target['img'], caption="TARGET SKETCH", use_container_width=True)
                for i, (idx, row) in enumerate(top_3.iterrows()):
                    with c_ai[i+1]:
                        st.image(row['image_url'], caption=f"Match: {row['score']:.1%}", use_container_width=True)
                        if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}", use_container_width=True):
                            st.session_state['sel'] = row.to_dict()

            # --- DETAILED COMPARISON ---
            selected = st.session_state.get('sel')
            if selected:
                st.divider()
                st.success(f"🔍 Comparing Detail with: **{selected['file_name']}**")
                if target.get("all_specs"):
                    st.subheader("📋 Measurement Comparison (Fuzzy Matched)")
                    all_exp = []
                    for sz, t_specs in target['all_specs'].items():
                        with st.expander(f"SIZE: {sz}", expanded=True):
                            r_specs = selected['spec_json'].get(sz, {})
                            rows = []; r_poms = list(r_specs.keys())
                            for t_pom, t_val in t_specs.items():
                                match = get_close_matches(t_pom, r_poms, n=1, cutoff=0.6)
                                rv = r_specs.get(match[0], 0) if match else 0
                                rows.append({"Measurement Point": t_pom, "Target": t_val, "Reference": rv, "Diff": f"{t_val-rv:+.3f}"})
                                all_exp.append({"Size": sz, "Point": t_pom, "Target": t_val, "Reference": rv, "Diff": t_val-rv})
                            st.table(pd.DataFrame(rows))
                    
                    buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                    pd.DataFrame(all_exp).to_excel(wr, index=False); wr.close()
                    st.download_button("📥 DOWNLOAD AUDIT REPORT", buf.getvalue(), f"Audit_{selected['file_name']}.xlsx", use_container_width=True)
                else: st.warning("⚠️ No measurement data found in Target file.")
