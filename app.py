import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
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

# ================= 2. AI CORE & HELPERS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # Cắt lề để AI tập trung vào hình vẽ, tránh nhiễu bảng biểu
    w, h = img.size; img = img.crop((w*0.05, h*0.05, w*0.95, h*0.7))
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(float).tolist() if norm > 0 else vec.tolist()

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower()
        if not t or any(x in t for x in ["wash", "color", "label", "style", "page"]): return 0
        # BỎ QUA MÃ POM (B101, D200...) ĐỂ KHÔNG BỊ SỐ TO
        if re.match(r'^[a-z]\d+', t): return 0
        t = t.replace(',', '.')
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        if num:
            val = float(num[0])
            return val if val < 150 else 0 # Chặn số rác > 150
        return 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()

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
                        pom_name = re.sub(r'[\d./\s]+$', '', line_txt).strip()
                        if len(pom_name) > 3:
                            for i, val in enumerate(line_vals):
                                s_key = f"Size_{i+1}"
                                if s_key not in all_specs: all_specs[s_key] = {}
                                all_specs[s_key][pom_name] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Models in Repo", f"{res_count.count or 0} SKUs")
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sync_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        for f in new_files:
            fb = f.getvalue(); data = extract_data(fb)
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "vector": get_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.session_state['up_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    file_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if file_audit:
        target = extract_data(file_audit.getvalue())
        if target and target['img']:
            # PHÂN TÍCH HÌNH DÁNG (AO RA AO, QUAN RA QUAN)
            img_obj = Image.open(io.BytesIO(target['img']))
            tw, th = img_obj.size
            is_long = (th / tw) > 1.4 # Tỷ lệ quần dài
            
            res = supabase.table("ai_data").select("id, vector, file_name").execute()
            if res.data:
                t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                valid_rows = []
                for r in res.data:
                    if r['vector'] and len(r['vector']) == 512:
                        name = r['file_name'].upper()
                        bonus = 0
                        # Thưởng điểm nếu đúng loại dựa trên tên file và hình dáng
                        if is_long and any(x in name for x in ["PANT", "JEAN", "TROUSER"]): bonus = 0.5
                        if not is_long and any(x in name for x in ["SHORT", "TOP", "SHIRT", "TEE"]): bonus = 0.5
                        r['sim_score'] = cosine_similarity(t_vec, np.array(r['vector']).reshape(1,-1))[0][0] + bonus
                        valid_rows.append(r)
                
                df_db = pd.DataFrame(valid_rows)
                top_3 = df_db.sort_values('sim_score', ascending=False).head(3)
                
                st.subheader("🎯 AI Matches")
                cols = st.columns(4)
                cols[0].image(target['img'], caption="TARGET PDF", use_container_width=True)
                for i, (idx, row) in enumerate(top_3.iterrows()):
                    det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).execute().data
                    if det:
                        with cols[i+1]:
                            st.image(det[0]['image_url'], caption=f"Match: {min(row['sim_score'], 1.0):.1%}")
                            if st.button(f"SELECT {i+1}", key=f"s_{idx}", use_container_width=True):
                                st.session_state['sel_audit'] = {**row.to_dict(), **det[0]}

            sel = st.session_state['sel_audit']
            if sel:
                st.divider(); st.success(f"📈 Comparing with: **{sel['file_name']}**")
                audit_dfs, sheet_names = [], []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        m_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                        r_specs = sel['spec_json'].get(m_sz[0], {}) if m_sz else {}
                        rows = []
                        for p, v in t_specs.items():
                            m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(m_p[0], 0) if m_p else 0
                            rows.append({"Point": p, "Target": v, "Ref": rv, "Diff": f"{v-rv:+.3f}"})
                        df_sz = pd.DataFrame(rows); st.table(df_sz)
                        audit_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Export Audit Excel", to_excel(audit_dfs, sheet_names), f"Audit_{sel['file_name']}.xlsx")

elif mode == "🔄 Version Control":
    st.subheader("🔄 Compare Two PDF Files")
    c1, c2 = st.columns(2)
    f1, f2 = c1.file_uploader("Version A:", type="pdf", key="v1"), c2.file_uploader("Version B:", type="pdf", key="v2")
    if f1 and f2:
        if st.button("⚡ Start Comparison", use_container_width=True):
            d1, d2 = extract_data(f1.getvalue()), extract_data(f2.getvalue())
            if d1 and d2:
                st.divider(); col_a, col_b = st.columns(2)
                col_a.image(d1['img'], caption="Ver A", use_container_width=True)
                col_b.image(d2['img'], caption="Ver B", use_container_width=True)
                all_sz = sorted(list(set(d1['all_specs'].keys()) | set(d2['all_specs'].keys())))
                version_dfs, ver_sheets = [], []
                for sz in all_sz:
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        s1, s2 = d1['all_specs'].get(sz, {}), d2['all_specs'].get(sz, {})
                        rows = []
                        for p in sorted(list(set(s1.keys()) | set(s2.keys()))):
                            v1, v2 = s1.get(p, 0), s2.get(p, 0)
                            diff = v2 - v1
                            rows.append({"Point": p, "Ver A": v1, "Ver B": v2, "Diff": f"{diff:+.3f}", "Status": "✅" if diff==0 else "⚠️"})
                        df_sz = pd.DataFrame(rows); st.table(df_sz)
                        version_dfs.append(df_sz); ver_sheets.append(sz)
                st.download_button("📥 Export Comparison Excel", to_excel(version_dfs, ver_sheets), "Version_Comparison.xlsx")
