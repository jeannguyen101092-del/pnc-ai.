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
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad(): 
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()
            return vec
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
        if num:
            val = float(num[0]) # SỬA LỖI: Lấy phần tử đầu tiên của list
            return val if val < 250 else 0
        return 0
    except: return 0

# ================= 3. PPJ COORDINATE SCRAPER =================
def extract_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
    try:
        # Load PDF for Image
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_pil.save(buf, format="WEBP", quality=70)
        img_bytes = buf.getvalue()
        doc.close()

        # Load PDF for Data
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
    except Exception as e:
        st.error(f"Lỗi phân tích PDF: {e}")
        return None

# ================= 4. SIDEBAR (SYNC LOGIC) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        count = res_count.count or 0
    except: count = 0
    
    st.metric("Models in Repo", f"{count} SKUs")
    st.write(f"💾 **Storage:** {count*0.08:.1f}MB / 1024MB")
    st.progress(min(count*0.08/1024, 1.0))
    st.divider()
    
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sync_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        logs = []
        progress_bar = st.progress(0)
        for idx, f in enumerate(new_files):
            try:
                fb = f.getvalue()
                data = extract_data(fb)
                if data and data['img']:
                    path = f"lib_{hashlib.md5(fb).hexdigest()}.webp"
                    # Upload Image
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    # AI Vector
                    vector = get_vector(data['img'])
                    
                    # Upsert Database
                    supabase.table("ai_data").upsert({
                        "id": f.name, 
                        "file_name": f.name, 
                        "vector": vector,
                        "spec_json": data['all_specs'], 
                        "image_url": img_url
                    }).execute()
                    logs.append({"File": f.name, "Status": "Success"})
                else:
                    logs.append({"File": f.name, "Status": "Failed (No Data)"})
            except Exception as e:
                logs.append({"File": f.name, "Status": f"Error: {str(e)}"})
            progress_bar.progress((idx + 1) / len(new_files))
            
        st.session_state['sync_results'] = logs
        st.session_state['up_key'] += 1
        st.rerun()

    if st.session_state['sync_results']:
        st.table(pd.DataFrame(st.session_state['sync_results']))
        if st.button("Clear Report"): 
            st.session_state['sync_results'] = None
            st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    file_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if file_audit:
        target = extract_data(file_audit.getvalue())
        if target and target['img']:
            # Lấy data từ DB để so sánh
            res = supabase.table("ai_data").select("id, vector, file_name").execute()
            all_db = res.data
            
            if all_db:
                df_db = pd.DataFrame(all_db)
                t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
                db_vecs = np.array([v for v in df_db['vector']])
                
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

                sel = st.session_state['sel_audit']
                if sel:
                    st.divider()
                    st.success(f"📈 Comparing with: **{sel['file_name']}**")
                    
                    for sz, t_specs in target['all_specs'].items():
                        with st.expander(f"SIZE: {sz}", expanded=True):
                            m_sz = get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)
                            r_specs = sel['spec_json'].get(m_sz[0], {}) if m_sz else {}
                            
                            res_list = []
                            for p, v in t_specs.items():
                                m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                                rv = r_specs.get(m_p[0], 0) if m_p else 0
                                diff = v - rv
                                res_list.append({"Point": p, "Target": v, "Ref": rv, "Diff": f"{diff:+.3f}"})
                            st.table(pd.DataFrame(res_list))
            else:
                st.warning("Kho dữ liệu trống. Vui lòng Sync dữ liệu ở Sidebar trước.")
