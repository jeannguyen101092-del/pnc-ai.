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
if 'sel_a' not in st.session_state: st.session_state['sel_a'] = None

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
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. AI TRANSLATION LOGIC =================
def translate_insight(text):
    mapping = {
        "WASH": "Hướng dẫn Giặt", "FABRIC": "Thành phần Vải", "STITCH": "Quy cách May",
        "THREAD": "Chỉ may", "LABEL": "Nhãn mác", "COLOR": "Màu sắc",
        "POCKET": "Chi tiết Túi", "WAIST": "Chi tiết Lưng/Cạp"
    }
    translated_notes = []
    lines = list(set([l.strip() for l in text.split('\n') if len(l.strip()) > 5]))
    for line in lines:
        for eng, vni in mapping.items():
            if eng in line.upper():
                translated_notes.append(f"**[{vni}]**: {line}")
                break
    return "\n\n".join(translated_notes[:10])

# ================= 4. PDF EXTRACTION =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary = {}, None, ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        img_temp = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_temp.save(buf, format="WEBP", quality=70)
        img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_text = ""
            for p in pdf.pages:
                full_text += (p.extract_text() or "") + "\n"
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    char_counts = df.apply(lambda x: x.astype(str).str.len().mean())
                    desc_col = char_counts.idxmax()
                    for col_idx in range(len(df.columns)):
                        if col_idx == desc_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, col_idx].head(15)]) > 0:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ')
                            if any(k in s_name.upper() for k in ["TOL", "GRADE", "SPEC"]): continue
                            temp_data = {str(df.iloc[d, desc_col]).strip(): parse_val(df.iloc[d, col_idx]) 
                                         for d in range(len(df)) if len(str(df.iloc[d, desc_col])) > 3}
                            if temp_data: all_specs[s_name] = temp_data
            vi_summary = translate_insight(full_text)
        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary}
    except: return None

# ================= 5. UI SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Repository Models", f"{count} SKUs")
    st.write(f"💾 **Storage:** {count * 0.06:.1f}MB / 1024MB")
    st.progress(min((count / 10000), 1.0))
    
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    c1, c2 = st.columns(2)
    with c1:
        if new_files and st.button("SYNCHRONIZE", use_container_width=True):
            with st.spinner("AI Processing..."):
                for f in new_files:
                    fb = f.read(); h = get_file_hash(fb)
                    check = supabase.table("ai_data").select("id").eq("id", h).execute()
                    if check.data: continue
                    data = extract_full_techpack(fb)
                    if data and data.get('img') and data.get('all_specs'):
                        path = f"lib_{h}.webp"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                        supabase.table("ai_data").upsert({
                            "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                            "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], 
                            "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                        }).execute()
            st.rerun()
    with c2:
        if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 6. MAIN AUDIT INTERFACE =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Select Mode:", ["Audit vs Repository", "Version Control (Compare 2 Rounds)"], horizontal=True)

if mode == "Audit vs Repository":
    file_audit = st.file_uploader("Upload PDF to Audit", type="pdf")
    if file_audit:
        target = extract_full_techpack(file_audit.read())
        if target:
            with st.expander("📝 AI TECHNICAL INSIGHT (TIẾNG VIỆT)", expanded=True):
                st.write(target['summary_vi'] if target['summary_vi'] else "Không tìm thấy ghi chú.")
            
            res = supabase.table("ai_data").select("*").execute()
            df_db = pd.DataFrame(res.data)
            
            t_name = file_audit.name.upper()
            def cat_filter(row_name):
                row_name = str(row_name).upper()
                is_t_pant = any(x in t_name for x in ["PANT", "QUAN", "SHORT"])
                is_r_pant = any(x in row_name for x in ["PANT", "QUAN", "SHORT"])
                return 1.3 if is_t_pant == is_r_pant else 0.0

            df_db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in df_db['vector']])).flatten()
            df_db['final'] = df_db['sim'] * df_db['file_name'].apply(cat_filter)
            top_3 = df_db.sort_values('final', ascending=False).head(3)
            
            st.subheader("🎯 Best Matches")
            cols = st.columns(4)
            cols[0].image(target['img'], caption="TARGET SKETCH", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i+1]:
                    st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}")
                    if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}"): st.session_state['sel_a'] = row.to_dict()

            selected = st.session_state.get('sel_a')
            if selected:
                st.success(f"Comparing with: {selected['file_name']}")
                all_export_data = []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = selected['spec_json'].get(sz, {})
                        rows = []
                        for p, v in t_specs.items():
                            match_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                            rv = r_specs.get(match_p[0], 0) if match_p else 0
                            diff = v - rv
                            rows.append({"Measurement Point": p, "Target": v, "Reference": rv, "Diff": f"{diff:+.3f}"})
                            all_export_data.append({"Size": sz, "Point": p, "Target": v, "Reference": rv, "Diff": diff})
                        st.table(pd.DataFrame(rows))
                
                # --- EXCEL EXPORT ---
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    pd.DataFrame(all_export_data).to_excel(writer, index=False, sheet_name='Comparison')
                st.download_button("📥 DOWNLOAD AUDIT REPORT (.XLSX)", buf.getvalue(), f"Audit_{selected['file_name']}.xlsx", use_container_width=True)

else: # Mode: Version Control
    st.subheader("🔄 Version Control: Compare 2 Style Rounds")
    res = supabase.table("ai_data").select("file_name, spec_json, image_url").execute()
    df_repo = pd.DataFrame(res.data)
    c1, c2 = st.columns(2)
    with c1:
        v_a = st.selectbox("Select Version A:", df_repo['file_name'].tolist(), key="va")
        data_a = df_repo[df_repo['file_name'] == v_a].iloc[0]
        st.image(data_a['image_url'], width=300)
    with c2:
        v_b = st.selectbox("Select Version B:", df_repo['file_name'].tolist(), key="vb")
        data_b = df_repo[df_repo['file_name'] == v_b].iloc[0]
        st.image(data_b['image_url'], width=300)
    
    if st.button("RUN ROUND COMPARISON", use_container_width=True):
        all_round_data = []
        for sz, specs_a in data_a['spec_json'].items():
            with st.expander(f"SIZE: {sz}", expanded=True):
                specs_b = data_b['spec_json'].get(sz, {})
                rows = []
                for p, v_a in specs_a.items():
                    v_b = specs_b.get(p, 0)
                    diff = v_b - v_a
                    rows.append({"Point": p, "Round A": v_a, "Round B": v_b, "Diff": f"{diff:+.3f}"})
                    all_round_data.append({"Size": sz, "Point": p, "Round A": v_a, "Round B": v_b, "Diff": diff})
                st.table(pd.DataFrame(rows))
        
        # --- EXCEL EXPORT FOR VERSION CONTROL ---
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            pd.DataFrame(all_round_data).to_excel(writer, index=False, sheet_name='Round_Comparison')
        st.download_button("📥 DOWNLOAD ROUND REPORT (.XLSX)", buf.getvalue(), f"Round_Comp_{data_a['file_name']}.xlsx", use_container_width=True)
