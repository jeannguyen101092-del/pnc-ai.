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
    if not img_bytes: return None
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower()
        if not t or any(x in t for x in ["wash", "color", "label", "style", "page", "date"]): return 0
        t = t.replace(',', '.')
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        val = float(num[0]) if num else 0
        return val if val < 200 else 0
    except: return 0

# ================= 3. CLEAN PDF EXTRACTION =================
def extract_techpack(file_content):
    if not file_content: return None
    all_specs, img_bytes, category = {}, None, "UNKNOWN"
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
    
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        # Capture full page sketch (More stable)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_temp = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_temp.save(buf, format="WEBP", quality=70); img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for page in pdf.pages:
                txt_data = page.extract_text() or ""
                full_txt += txt_data + "\n"
                tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 4})
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    desc_col = -1
                    for c_idx in range(len(df.columns)):
                        col_str = " ".join(df.iloc[:, c_idx].astype(str).upper())
                        if any(kw in col_str for kw in POM_KWS): desc_col = c_idx; break
                    
                    if desc_col != -1:
                        for col_idx in range(len(df.columns)):
                            if col_idx == desc_col: continue
                            vals = [parse_val(v) for v in df.iloc[:, col_idx]]
                            if sum(1 for v in vals if v > 0) >= 3:
                                s_name = str(df.iloc[0, col_idx]).strip() or f"Size_{col_idx}"
                                if s_name not in all_specs: all_specs[s_name] = {}
                                for d_idx in range(len(df)):
                                    pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                    val = parse_val(df.iloc[d_idx, col_idx])
                                    if val > 0 and len(pom) > 3: all_specs[s_name][pom] = val
            
            category = "BOTTOM" if any(x in full_txt.upper() for x in ["PANT", "QUAN", "SHORT"]) else "TOP"
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR (STORAGE DISPLAY) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Repository Models", f"{count} SKUs")
    
    used_mb = (count * 0.08) 
    st.write(f"💾 **Cloud Storage:** {used_mb:.1f}MB / 1024MB")
    st.progress(min(used_mb / 1024, 1.0))

    st.divider()
    new_files = st.file_uploader("Sync to Repository", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE", use_container_width=True):
        with st.spinner("Processing..."):
            for f in new_files:
                fb = f.getvalue(); h = get_file_hash(fb)
                data = extract_techpack(fb)
                if data and data['img']:
                    path = f"lib_{h}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "category": data['category'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
        st.rerun()
    if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode:", ["🔍 AI Search (Audit)", "🔄 Version Control"], horizontal=True)

if mode == "🔍 AI Search (Audit)":
    file_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if file_audit:
        target = extract_techpack(file_audit.getvalue())
        if target and target.get('img'):
            all_db = []
            for i in range(0, count, 1000):
                res = supabase.table("ai_data").select("id, vector, file_name, category").range(i, i + 999).execute()
                all_db.extend(res.data)
            df_db = pd.DataFrame(all_db)
            
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            
            # Smart Filter
            t_cat = target['category']
            def get_score(row):
                if t_cat in ["TOP", "BOTTOM"] and row.get('category') in ["TOP", "BOTTOM"] and t_cat != row.get('category'): return 0.0
                return 1.3 if t_cat == row.get('category') else 1.0
            
            df_db['final'] = df_db['sim'] * df_db.apply(get_score, axis=1)
            top_3 = df_db.sort_values('final', ascending=False).head(3)

            st.subheader("🎯 Best Matches")
            cols = st.columns(4)
            cols.image(target['img'], caption="TARGET", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).single().execute()
                with cols[i+1]:
                    st.image(det.data['image_url'], caption=f"Match: {row['sim']:.1%}")
                    if st.button(f"SELECT {i+1}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['sel_audit'] = {**row.to_dict(), **det.data}

            selected = st.session_state.get('sel_audit')
            if selected:
                st.divider()
                st.success(f"Comparing with: **{selected['file_name']}**")
                all_ex = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_sz = get_close_matches(sz, list(selected['spec_json'].keys()), 1, 0.4)
                        r_specs = selected['spec_json'].get(r_sz[0] if r_sz else "", {})
                        rows = []
                        for p, v in t_specs.items():
                            m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(m_p[0] if m_p else "", 0)
                            rows.append({"Point": p, "Target": v, "Ref": rv, "Diff": f"{v-rv:+.3f}"})
                            all_ex.append({"Size": sz, **rows[-1]})
                        st.table(pd.DataFrame(rows))
                if all_ex:
                    buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                    pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                    st.download_button("📥 DOWNLOAD AUDIT", buf.getvalue(), f"Audit.xlsx", use_container_width=True)

else: # Mode Version Control
    st.subheader("🔄 Compare Round A (Repo) vs Round B (New Upload)")
    all_n = []
    for i in range(0, count, 1000):
        res_n = supabase.table("ai_data").select("file_name").range(i, i+999).execute()
        all_n.extend([r['file_name'] for r in res_n.data])
    
    c1, c2 = st.columns(2)
    with c1:
        v_a_name = st.selectbox("Style A (Repo):", all_n)
        res_a = supabase.table("ai_data").select("*").eq("file_name", v_a_name).single().execute()
        data_a = res_a.data
        if data_a: st.image(data_a['image_url'], width=350)
    with c2:
        file_b = st.file_uploader("Style B (New Round):", type="pdf", key="v_fb")
        if file_b:
            data_b = extract_techpack(file_b.getvalue())
            if data_b and data_b.get('img'): st.image(data_b['img'], width=350)

    if file_b and data_b and data_a:
        st.divider()
        all_r_ex = []
        for sz, specs_b in data_b['all_specs'].items():
            with st.expander(f"SIZE: {sz}", expanded=True):
                r_sz = get_close_matches(sz, list(data_a['spec_json'].keys()), 1, 0.4)
                specs_a = data_a['spec_json'].get(r_sz[0] if r_sz else "", {})
                rows = []
                for p_b, v_b in specs_b.items():
                    m_p = get_close_matches(p_b, list(specs_a.keys()), 1, 0.6)
                    v_a = specs_a.get(m_p[0] if m_p else "", 0)
                    rows.append({"Point": p_b, "Round A": v_a, "Round B": v_b, "Diff": f"{v_b-v_a:+.3f}"})
                    all_r_ex.append({"Size": sz, **rows[-1]})
                st.table(pd.DataFrame(rows))
        if all_r_ex:
            buf_r = io.BytesIO(); wr = pd.ExcelWriter(buf_r, engine='xlsxwriter')
            pd.DataFrame(all_r_ex).to_excel(wr, index=False); wr.close()
            st.download_button("📥 DOWNLOAD COMPARISON", buf_r.getvalue(), f"Round_Comparison.xlsx", use_container_width=True)
