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

# ================= 2. AI CORE (IMPROVED & STABLE) =================
@st.cache_resource
def load_model():
    # Sử dụng ResNet18 để đảm bảo tốc độ và tương thích 512 chiều
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
    model.eval()
    return model

model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        # Xử lý ảnh: Chuyển về RGB và Resize chuẩn
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
            # CHUẨN HÓA VECTOR: Giúp so sánh dáng Sketch chính xác hơn rất nhiều
            norm = np.linalg.norm(vec)
            if norm > 0: vec = vec / norm
            return vec.astype(float).tolist()
    except: return None

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
        return float(num[0]) if num else 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()

# ================= 3. SCRAPER =================
def extract_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
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

# ================= 4. SIDEBAR (SYNC) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        count = res_count.count or 0
    except: count = 0
    st.metric("Models in Repo", f"{count} SKUs")
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sync_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        logs = []
        for f in new_files:
            try:
                fb = f.getvalue(); data = extract_data(fb)
                if data and data['img']:
                    file_hash = hashlib.md5(f.name.encode()).hexdigest()
                    valid_uuid = str(uuid.UUID(file_hash))
                    path = f"lib_{file_hash}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": valid_uuid, "file_name": f.name, "vector": get_vector(data['img']), 
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
                    logs.append({"File": f.name, "Status": "Success"})
                else: logs.append({"File": f.name, "Status": "Failed"})
            except Exception as e: logs.append({"File": f.name, "Status": str(e)})
        st.session_state['sync_results'] = logs
        st.session_state['up_key'] += 1; st.rerun()
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
            res = supabase.table("ai_data").select("id, vector, file_name").execute()
            all_db = res.data
            if all_db:
                # 1. Tạo Vector cho file target
                t_vec_raw = get_vector(target['img'])
                t_vec = np.array(t_vec_raw).reshape(1, -1)
                
                # 2. Lọc bỏ dữ liệu lỗi kích thước (Tránh lỗi ValueError)
                valid_data = [r for r in all_db if r['vector'] and len(r['vector']) == len(t_vec_raw)]
                
                if valid_data:
                    df_db = pd.DataFrame(valid_data)
                    db_vecs = np.array([v for v in df_db['vector']])
                    
                    # 3. So sánh tương đồng
                    df_db['sim'] = cosine_similarity(t_vec, db_vecs).flatten()
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
                else:
                    st.warning("Kho dữ liệu hiện tại không tương thích với Model AI mới. Vui lòng nhấn SYNCHRONIZE lại.")

                sel = st.session_state['sel_audit']
                if sel:
                    st.divider(); st.success(f"📈 Comparing: **{sel['file_name']}**")
                    all_dfs, sheet_names = [], []
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
                            all_dfs.append(df_sz); sheet_names.append(sz)
                    st.download_button("📥 Download Excel Report", to_excel(all_dfs, sheet_names), f"Audit_{sel['file_name']}.xlsx")

elif mode == "🔄 Version Control":
    st.subheader("🔄 Compare Two New Versions")
    c_up1, c_up2 = st.columns(2)
    f1 = c_up1.file_uploader("Upload Version A (Old):", type="pdf", key="va")
    f2 = c_up2.file_uploader("Upload Version B (New):", type="pdf", key="vb")
    if f1 and f2:
        if st.button("⚡ Start Visual & Spec Comparison", use_container_width=True):
            d1, d2 = extract_data(f1.getvalue()), extract_data(f2.getvalue())
            if d1 and d2:
                st.divider(); im1, im2 = st.columns(2)
                im1.image(d1['img'], caption=f"Ver A: {f1.name}", use_container_width=True)
                im2.image(d2['img'], caption=f"Ver B: {f2.name}", use_container_width=True)
                st.divider(); st.markdown("### 📊 Full Spec Comparison")
                all_sz = sorted(list(set(d1['all_specs'].keys()) | set(d2['all_specs'].keys())))
                all_dfs, sheet_names = [], []
                for sz in all_sz:
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        s1, s2 = d1['all_specs'].get(sz, {}), d2['all_specs'].get(sz, {})
                        all_p = sorted(list(set(s1.keys()) | set(s2.keys())))
                        res = []
                        for p in all_p:
                            v1, v2 = s1.get(p, 0), s2.get(p, 0)
                            diff = v2 - v1
                            stt = "✅" if diff == 0 else "⚠️"
                            if v1 == 0: stt = "🆕"
                            if v2 == 0: stt = "❌"
                            res.append({"Point": p, "Ver A": v1, "Ver B": v2, "Diff": f"{diff:+.3f}", "Status": stt})
                        df_sz = pd.DataFrame(res); st.table(df_sz)
                        all_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Download Excel Comparison", to_excel(all_dfs, sheet_names), "Comparison_Report.xlsx")
