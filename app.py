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
    mapping = {"WASH": "HD Giặt", "FABRIC": "Thành phần Vải", "STITCH": "Quy cách May", "LABEL": "Nhãn mác", "COLOR": "Màu sắc", "POCKET": "Túi", "WAIST": "Lưng/Cạp"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

# ================= 3. PDF EXTRACTION (ULTRA-CLEAR) =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary = {}, None, ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0)) # Super Clear
        img_t = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_t.save(buf, format="WEBP", quality=75); img_bytes = buf.getvalue()
        doc.close()
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for p in pdf.pages:
                full_txt += (p.extract_text() or "") + "\n"
                for tb in p.extract_tables():
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    d_col = df.apply(lambda x: x.astype(str).str.len().mean()).idxmax()
                    for c_idx in range(len(df.columns)):
                        if c_idx == d_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, c_idx].head(15)]) > 0:
                            s_n = str(df.iloc[0, c_idx]).strip().replace('\n', ' ')
                            if any(k in s_n.upper() for k in ["TOL", "GRADE", "SPEC"]): continue
                            specs = {str(df.iloc[d, d_col]).strip(): parse_val(df.iloc[d, c_idx]) for d in range(len(df)) if len(str(df.iloc[d, d_col])) > 3}
                            if specs: all_specs[s_n] = specs
            vi_summary = translate_insight(full_txt)
        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary}
    except: return None

# ================= 4. SIDEBAR (STORAGE) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Repository Models", f"{count} SKUs")
    st.write(f"💾 **Storage:** {count * 0.07:.1f}MB / 1024MB")
    st.progress(min((count / 10000), 1.0))
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    c1, c2 = st.columns(2)
    with c1:
        if new_files and st.button("SYNCHRONIZE", use_container_width=True):
            with st.spinner("AI Saving..."):
                for f in new_files:
                    fb = f.read(); h = get_file_hash(fb)
                    data = extract_full_techpack(fb)
                    if data and data.get('img') and data.get('all_specs'):
                        path = f"lib_{h}.webp"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                        supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
            st.rerun()
    with c2:
        if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Working Mode:", ["🔍 AI Similarity Search (Audit)", "🔄 Version Control (Repo vs Upload)"], horizontal=True)

if mode == "🔍 AI Similarity Search (Audit)":
    file_audit = st.file_uploader("Upload PDF to find matches in Repository:", type="pdf")
    if file_audit:
        target = extract_full_techpack(file_audit.read())
        if target:
            with st.expander("📝 AI TECHNICAL INSIGHT (TIẾNG VIỆT)", expanded=True):
                st.write(target['summary_vi'] if target['summary_vi'] else "No special notes found.")
            
            res = supabase.table("ai_data").select("*").execute()
            df_db = pd.DataFrame(res.data)
            
            # --- AI SENSITIVITY LOGIC (GIỮ NGUYÊN) ---
            t_name = file_audit.name.upper()
            def cat_filter(row_name):
                row_name = str(row_name).upper()
                is_t_pant = any(x in t_name for x in ["PANT", "QUAN", "SHORT"])
                is_r_pant = any(x in row_name for x in ["PANT", "QUAN", "SHORT"])
                if is_t_pant != is_r_pant: return 0.0
                bonus = 1.0
                for kw in ["CARGO", "THUN", "ELASTIC", "WAIST"]: 
                    if (kw in t_name) == (kw in row_name): bonus += 0.2
                return bonus

            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            df_db['final'] = df_db['sim'] * df_db['file_name'].apply(cat_filter)
            top_3 = df_db.sort_values('final', ascending=False).head(3)
            
            st.subheader("🎯 Best AI Matches (Quick Search)")
            cols = st.columns(4)
            with cols[0]: st.image(target['img'], caption="TARGET FILE", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i+1]:
                    st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}")
                    if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}"): st.session_state['sel_audit'] = row.to_dict()

            selected = st.session_state.get('sel_audit')
            if selected:
                st.divider()
                st.success(f"Comparing with: {selected['file_name']}")
                all_ex = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = selected['spec_json'].get(sz, {})
                        rows = []
                        for p, v in t_specs.items():
                            m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(m_p[0], 0) if m_p else 0
                            rows.append({"Measurement Point": p, "Target": v, "Reference": rv, "Diff": f"{v-rv:+.3f}"})
                            all_ex.append({"Size": sz, "Point": p, "Target": v, "Reference": rv, "Diff": v-rv})
                        st.table(pd.DataFrame(rows))
                
                buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                st.download_button("📥 DOWNLOAD AUDIT REPORT (.XLSX)", buf.getvalue(), f"Audit_{selected['file_name']}.xlsx", use_container_width=True)

else: # --- MODE: VERSION CONTROL (A FROM REPO VS B FROM UPLOAD) ---
    st.subheader("🔄 Compare Round A (From Repo) vs Round B (New Upload)")
    res = supabase.table("ai_data").select("file_name, spec_json, image_url").execute()
    df_repo = pd.DataFrame(res.data)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Step 1: Select Reference Style (Round A)")
        ver_a_name = st.selectbox("Search Repo SKU:", df_repo['file_name'].tolist(), key="va")
        data_a = df_repo[df_repo['file_name'] == ver_a_name].iloc[0]
        st.image(data_a['image_url'], caption=f"Stored: {ver_a_name}", width=350)
    with c2:
        st.info("Step 2: Upload New Version (Round B)")
        file_b = st.file_uploader("Upload New Tech-Pack:", type="pdf", key="file_b")
        if file_b:
            data_b = extract_full_techpack(file_b.read())
            if data_b:
                st.image(data_b['img'], caption="Newly Uploaded Style", width=350)

    if file_b and st.button("RUN DIRECT COMPARISON", use_container_width=True):
        st.divider()
        st.write(f"### 📊 Comparing **{ver_a_name}** vs **{file_b.name}**")
        all_round_data = []
        for sz, specs_b in data_b['all_specs'].items():
            with st.expander(f"SIZE: {sz}", expanded=True):
                specs_a = data_a['spec_json'].get(sz, {})
                rows = []
                for p_b, v_b in specs_b.items():
                    m_p = get_close_matches(p_b, list(specs_a.keys()), 1, 0.6)
                    v_a = specs_a.get(m_p[0], 0) if m_p else 0
                    rows.append({"Point": p_b, "Stored (A)": v_a, "New (B)": v_b, "Diff": f"{v_b-v_a:+.3f}"})
                    all_round_data.append({"Size": sz, "Point": p_b, "Stored_A": v_a, "New_B": v_b, "Diff": v_b-v_a})
                st.table(pd.DataFrame(rows))
        
        buf_r = io.BytesIO(); wr = pd.ExcelWriter(buf_r, engine='xlsxwriter')
        pd.DataFrame(all_round_data).to_excel(wr, index=False); wr.close()
        st.download_button("📥 DOWNLOAD COMPARISON (.XLSX)", buf_r.getvalue(), f"Round_Comp_{ver_a_name}.xlsx", use_container_width=True)
