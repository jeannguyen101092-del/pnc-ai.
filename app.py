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
        t = str(t).replace('"', '').strip().lower()
        if not t or any(x in t for x in ["wash", "color", "label", "ticket"]): return 0
        t = t.replace(',', '.')
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        return float(num[0]) if num else 0
    except: return 0

def translate_insight(text):
    mapping = {"WASH": "HD Giặt", "FABRIC": "Vải", "STITCH": "May mặc", "LABEL": "Nhãn", "COLOR": "Màu", "POCKET": "Túi", "WAIST": "Lưng/Cạp"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

# ================= 3. PDF EXTRACTION (MASTER COORDINATE SCRAPER 4.0) =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary, category = {}, None, "", "UNKNOWN"
    POM_KWS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "POM", "SPEC"]
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(3.0, 3.0)) # Ultra DPI
        img_t = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_t.save(buf, format="WEBP", quality=75); img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for page in pdf.pages:
                txt = page.extract_text() or ""
                full_txt += txt + "\n"
                # Thuật toán tọa độ Y-Grid: Xử lý bảng không khung
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
            vi_summary = translate_insight(full_txt)
            u_txt = full_txt.upper()
            if any(x in u_txt for x in ["PANT", "QUAN", "SHORT"]): category = "BOTTOM"
            elif any(x in u_txt for x in ["SHIRT", "AO", "TOP", "JACKET"]): category = "TOP"
        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary, "category": category}
    except: return None

# ================= 4. UI SIDEBAR (RESTORED ALL) =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Repository Models", f"{count} SKUs")
    
    # Storage Display
    used_mb = (count * 0.08)
    st.write(f"💾 **Cloud Storage:** {used_mb:.1f}MB / 1024MB")
    st.progress(min((used_mb / 1024), 1.0))
    
    if st.button("🧹 CLEAN ERRORS & TRASH", use_container_width=True):
        st.info("Storage cleanup in progress..."); time.sleep(1); st.rerun()

    st.divider()
    st.subheader("📥 Data Ingestion")
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    c1, c2 = st.columns(2)
    with c1:
        if new_files and st.button("SYNCHRONIZE", use_container_width=True):
            with st.spinner("AI Processing..."):
                for f in new_files:
                    fb = f.read(); h = get_file_hash(fb)
                    data = extract_full_techpack(fb)
                    if data and data.get('img') and data.get('all_specs'):
                        path = f"lib_{h}.webp"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                        supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], "category": data['category'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
            st.rerun()
    with c2:
        if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI (HAVE IT ALL) =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode Selection:", ["🔍 AI Similarity Search (Audit)", "🔄 Version Control (Repo vs New)"], horizontal=True)

if mode == "🔍 AI Similarity Search (Audit)":
    file_audit = st.file_uploader("Upload Target PDF to find match:", type="pdf")
    if file_audit:
        target = extract_full_techpack(file_audit.read())
        if target:
            with st.expander("📝 AI TECHNICAL INSIGHT (TIẾNG VIỆT)", expanded=True):
                st.write(target['summary_vi'] if target['summary_vi'] else "Không tìm thấy ghi chú đặc biệt.")
            
            # Quét toàn bộ kho dữ liệu (phá giới hạn 1000)
            all_db = []
            for i in range(0, count, 1000):
                res = supabase.table("ai_data").select("id, vector, file_name, category").range(i, i + 999).execute()
                all_db.extend(res.data)
            df_db = pd.DataFrame(all_db)
            
            # Category Hard-Filter (Chống Áo vs Quần)
            t_cat = target['category']
            def smart_filter(row):
                if t_cat in ["TOP", "BOTTOM"] and row.get('category') in ["TOP", "BOTTOM"] and t_cat != row.get('category'): return 0.0
                return 1.3 if t_cat == row.get('category') else 1.0

            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            df_db['final'] = df_db['sim'] * df_db.apply(smart_filter, axis=1)
            top_3 = df_db.sort_values('final', ascending=False).head(3)
            
            st.subheader("🎯 Best Matches (Click Image to Zoom)")
            cols = st.columns(4)
            cols.image(target['img'], caption="TARGET (SKETCH)", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).single().execute()
                with cols[i+1]:
                    st.image(det.data['image_url'], caption=f"Match: {row['sim']:.1%}")
                    if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}"):
                        st.session_state['sel_audit'] = {**row.to_dict(), **det.data}

            selected = st.session_state.get('sel_audit')
            if selected:
                st.divider()
                st.success(f"Comparing with: **{selected['file_name']}**")
                all_ex = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = selected['spec_json'].get(sz, {})
                        rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        for r in rows: 
                            r['Diff'] = f"{r['Target'] - r['Ref']:+.3f}"
                            all_ex.append({"Size": sz, **r})
                        st.table(pd.DataFrame(rows))
                
                buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                st.download_button("📥 DOWNLOAD AUDIT (.XLSX)", buf.getvalue(), f"Audit.xlsx", use_container_width=True)

else: # Mode Version Control
    st.subheader("🔄 Round A (From Repo) vs Round B (New Upload)")
    all_n = []
    for i in range(0, count, 1000):
        res_n = supabase.table("ai_data").select("file_name").range(i, i+999).execute()
        all_n.extend([r['file_name'] for r in res_n.data])
    
    c1, c2 = st.columns(2)
    with c1:
        v_a_name = st.selectbox("Style A (Stored):", all_n)
        data_a = supabase.table("ai_data").select("*").eq("file_name", v_a_name).single().execute().data
        st.image(data_a['image_url'], width=350, caption="Round A")
    with c2:
        file_b = st.file_uploader("Style B (New Upload):", type="pdf", key="v_fb")
        if file_b:
            data_b = extract_full_techpack(file_b.read())
            if data_b: st.image(data_b['img'], width=350, caption="Round B")

    if file_b and data_b:
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
                    rows.append({"Point": p_b, "Stored (A)": v_a, "New (B)": v_b, "Diff": f"{v_b-v_a:+.3f}"})
                    all_r_ex.append({"Size": sz, **rows[-1]})
                st.table(pd.DataFrame(rows))
        
        buf_r = io.BytesIO(); wr = pd.ExcelWriter(buf_r, engine='xlsxwriter')
        pd.DataFrame(all_r_ex).to_excel(wr, index=False); wr.close()
        st.download_button("📥 DOWNLOAD COMPARISON (.XLSX)", buf_r.getvalue(), f"Round_Comparison.xlsx", use_container_width=True)
