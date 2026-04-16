import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
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

# ================= 2. AI CORE =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower()
        if not t or any(x in t for x in ["wash", "color", "label", "style", "page"]): return 0
        t = t.replace(',', '.')
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        val = float(num[0]) if num else 0
        return val if val < 250 else 0
    except: return 0

# ================= 3. PPJ COORDINATE SCRAPER =================
def extract_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        buf = io.BytesIO(); Image.open(io.BytesIO(pix.tobytes("png"))).save(buf, format="WEBP", quality=70)
        img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = (df_w['top'] / 2).round() * 2
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text'])
                    line_vals = [parse_val(w) for w in line_txt.split() if parse_val(w) > 0]
                    if any(kw in line_txt.upper() for kw in POM_KWS) and line_vals:
                        pom_name = re.sub(r'[0-9./\s]+$', '', line_txt).strip()
                        if len(pom_name) > 3:
                            for i, val in enumerate(line_vals):
                                s_key = f"Size_{i+1}"
                                if s_key not in all_specs: all_specs[s_key] = {}
                                all_specs[s_key][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR (FIXED SYNC) =================
with st.sidebar:
    st.title("👔 PPJ GROUP")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Models in Repository", f"{count} SKUs")
    st.write(f"💾 **Cloud Storage:** {count*0.08:.1f}MB / 1024MB")
    st.progress(min(count*0.08/1024, 1.0))
    st.divider()
    
    new_files = st.file_uploader("Upload Tech-Packs to Sync", accept_multiple_files=True)
    if new_files and st.button("SYNCHRONIZE", use_container_width=True):
        success_num = 0
        with st.spinner("AI is storing data..."):
            for f in new_files:
                fb = f.getvalue(); h = hashlib.md5(fb).hexdigest()
                # Kiểm tra trùng
                check = supabase.table("ai_data").select("id").eq("id", h).execute()
                if check.data: continue
                
                data = extract_data(fb)
                if data and data['img']:
                    path = f"lib_{h}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
                    success_num += 1
        if success_num > 0:
            st.sidebar.success(f"✅ Added {success_num} SKU(s)")
            time.sleep(1); st.rerun()

# ================= 5. MAIN UI (AUDIT & VERSION CONTROL) =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["🔍 Audit Mode (Repo Search)", "🔄 Version Control (Repo vs Local)"], horizontal=True)

if mode == "🔍 Audit Mode (Repo Search)":
    file_audit = st.file_uploader("Upload Target PDF to Audit:", type="pdf")
    if file_audit:
        target = extract_data(file_audit.getvalue())
        if target and target['img']:
            # Fetch Repo (Phá giới hạn 1000)
            all_db = []
            for i in range(0, count, 1000):
                res = supabase.table("ai_data").select("id, vector, file_name").range(i, i+999).execute()
                all_db.extend(res.data)
            df_db = pd.DataFrame(all_db)
            
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)

            st.subheader("🎯 AI Best Matches")
            cols = st.columns(4)
            cols[0].image(target['img'], caption="TARGET PDF", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                res_det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute()
                if res_det.data:
                    det = res_det.data[0]
                    with cols[i+1]:
                        st.image(det['image_url'], caption=f"Match: {row['sim']:.1%}")
                        if st.button(f"SELECT {i+1}", key=f"btn_{idx}", use_container_width=True):
                            st.session_state['sel_audit'] = {**row.to_dict(), **det}

            sel = st.session_state['sel_audit']
            if sel:
                st.divider()
                st.success(f"📈 Comparing with: **{sel['file_name']}**")
                all_ex = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                        r_specs = sel['spec_json'].get(r_sz[0] if r_sz else "", {})
                        rows = []
                        for p, v in t_specs.items():
                            m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(m_p[0] if m_p else "", 0)
                            rows.append({"Point": p, "Target": v, "Reference": rv, "Diff": f"{v-rv:+.3f}"})
                            all_ex.append({"Size": sz, **rows[-1]})
                        st.table(pd.DataFrame(rows))
                if all_ex:
                    buf = io.BytesIO(); pd.DataFrame(all_ex).to_excel(pd.ExcelWriter(buf), index=False)
                    st.download_button("📥 DOWNLOAD AUDIT REPORT", buf.getvalue(), "Audit_Report.xlsx", use_container_width=True)

else: # --- MODE: VERSION CONTROL (REPO VS LOCAL) ---
    st.subheader("🔄 Compare Version A (Stored) vs Version B (New Upload)")
    # Lấy tên tất cả file (Phá 1000 limit)
    all_names = []
    for i in range(0, count, 1000):
        all_names.extend([r['file_name'] for r in supabase.table("ai_data").select("file_name").range(i, i+999).execute().data])
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Style A: Select from Repository")
        v_a_name = st.selectbox("Search Stored SKU:", all_names)
        res_a = supabase.table("ai_data").select("*").eq("file_name", v_a_name).execute()
        data_a = res_a.data[0] if res_a.data else None
        if data_a: st.image(data_a['image_url'], caption=f"Stored: {v_a_name}", use_container_width=True)
    
    with col2:
        st.info("Style B: Upload New PDF File")
        file_b = st.file_uploader("Upload New Tech-Pack:", type="pdf", key="fb_round")
        data_b = None
        if file_b:
            data_b = extract_data(file_b.getvalue())
            if data_b: st.image(data_b['img'], caption=f"New: {file_b.name}", use_container_width=True)

    if file_b and data_b and data_a:
        if st.button("RUN DIRECT COMPARISON", use_container_width=True):
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
                        diff = v_b - v_a
                        rows.append({"Point": p_b, "Stored (A)": v_a, "New (B)": v_b, "Diff": f"{diff:+.3f}"})
                        all_r_ex.append({"Size": sz, **rows[-1]})
                    st.table(pd.DataFrame(rows))
            if all_r_ex:
                buf_r = io.BytesIO(); pd.DataFrame(all_r_ex).to_excel(pd.ExcelWriter(buf_r), index=False)
                st.download_button("📥 DOWNLOAD COMPARISON (.XLSX)", buf_r.getvalue(), "Comparison_Report.xlsx", use_container_width=True)
