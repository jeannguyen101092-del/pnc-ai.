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
if 'sync_results' not in st.session_state: st.session_state['sync_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

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
        val = float(num) if num else 0
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
        img_bytes = buf.getvalue(); doc.close()

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

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Comparison')
    return output.getvalue()

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Models in Repo", f"{count} SKUs")
    st.write(f"💾 **Storage:** {count*0.08:.1f}MB / 1024MB")
    st.progress(min(count*0.08/1024, 1.0))
    st.divider()
    
    new_files = st.file_uploader("Upload Tech-Packs to Sync", accept_multiple_files=True, key=f"sync_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE & REPAIR", use_container_width=True):
        logs = []
        with st.spinner("AI is storing data..."):
            for f in new_files:
                fb = f.getvalue(); data = extract_data(fb)
                if data and data['img']:
                    path = f"lib_{hashlib.md5(fb).hexdigest()}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": f.name, "file_name": f.name, "vector": get_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
                    logs.append({"File": f.name, "Status": "Success"})
                else: logs.append({"File": f.name, "Status": "Failed"})
        st.session_state['sync_results'] = logs
        st.session_state['up_key'] += 1
        st.sidebar.success("Process Completed!"); time.sleep(1); st.rerun()

    if st.session_state['sync_results']:
        st.table(pd.DataFrame(st.session_state['sync_results']))
        if st.button("Clear Report"): st.session_state['sync_results'] = None; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    file_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if file_audit:
        target = extract_data(file_audit.getvalue())
        if target and target['img']:
            all_db = [r for i in range(0, count, 1000) for r in supabase.table("ai_data").select("id, vector, file_name").range(i, i+999).execute().data]
            df_db = pd.DataFrame(all_db)
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)

            st.subheader("🎯 AI Matches")
            cols = st.columns(4)
            cols[0].image(target['img'], caption="TARGET PDF", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute().data
                if det:
                    with cols[i+1]:
                        st.image(det[0]['image_url'], caption=f"Match: {row['sim']:.1%}")
                        if st.button(f"SELECT {i+1}", key=f"s_{idx}", use_container_width=True):
                            st.session_state['sel_audit'] = {**row.to_dict(), **det[0]}

            sel = st.session_state['sel_audit']
            if sel:
                st.divider()
                st.success(f"📈 Comparing with: **{sel['file_name']}**")
                final_ex = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        m_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                        r_specs = sel['spec_json'].get(m_sz[0], {}) if m_sz else {}
                        rows = []
                        for p, v in t_specs.items():
                            m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(m_p[0], 0) if m_p else 0
                            diff = v - rv
                            rows.append({"Point": p, "Target": v, "Ref": rv, "Diff": f"{diff:+.3f}"})
                            final_ex.append({"Size": sz, "Point": p, "Target": v, "Ref": rv, "Diff": diff})
                        st.table(pd.DataFrame(rows))
                if final_ex:
                    st.download_button("📥 DOWNLOAD AUDIT REPORT (.XLSX)", to_excel(pd.DataFrame(final_ex)), f"Audit_{sel['file_name']}.xlsx", use_container_width=True)

else: # Mode Version Control
    st.subheader("🔄 Compare Version A (Repo) vs Version B (Local)")
    all_n = list(set([r['file_name'] for i in range(0, count, 1000) for r in supabase.table("ai_data").select("file_name").range(i, i+999).execute().data]))
    
    col_a, col_b = st.columns(2)
    with col_a:
        v_a_name = st.selectbox("Style A (Repo):", all_n)
        res_a = supabase.table("ai_data").select("*").eq("file_name", v_a_name).execute()
        data_a = res_a.data[0] if res_a.data else None
        if data_a: st.image(data_a['image_url'], width=350, caption="Round A (Repo)")
    with col_b:
        file_b = st.file_uploader("Style B (New Upload):", type="pdf", key="fb_r")
        data_b = extract_data(file_b.getvalue()) if file_b else None
        if data_b: st.image(data_b['img'], width=350, caption="Round B (New)")

    if file_b and data_b and data_a:
        if st.button("RUN COMPARISON", use_container_width=True):
            st.divider()
            vc_ex = []
            for sz, specs_b in data_b['all_specs'].items():
                with st.expander(f"SIZE: {sz}", expanded=True):
                    m_sz = get_close_matches(sz, list(data_a['spec_json'].keys()), 1, 0.4)
                    specs_a = data_a['spec_json'].get(m_sz[0], {}) if m_sz else {}
                    rows = []
                    for p_b, v_b in specs_b.items():
                        m_p = get_close_matches(p_b, list(specs_a.keys()), 1, 0.6)
                        v_a = specs_a.get(m_p[0], 0) if m_p else 0
                        diff = v_b - v_a
                        rows.append({"Point": p_b, "Repo (A)": v_a, "New (B)": v_b, "Diff": f"{diff:+.3f}"})
                        vc_ex.append({"Size": sz, "Point": p_b, "Repo_A": v_a, "New_B": v_b, "Diff": diff})
                    st.table(pd.DataFrame(rows))
            if vc_ex:
                st.download_button("📥 DOWNLOAD COMPARISON (.XLSX)", to_excel(pd.DataFrame(vc_ex)), "Comparison_Report.xlsx", use_container_width=True)
