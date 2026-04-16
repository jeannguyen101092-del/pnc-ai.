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
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

def translate_insight(text):
    mapping = {"WASH": "HD Giặt", "FABRIC": "Vải", "STITCH": "May mặc", "LABEL": "Nhãn", "COLOR": "Màu", "POCKET": "Túi", "WAIST": "Lưng"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

# ================= 3. PDF EXTRACTION (SKETCH ISOLATION) =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary, category = {}, None, "", "UNKNOWN"
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        
        # --- CÔ LẬP HÌNH VẼ (BỎ BẢNG) ---
        paths = page.get_drawings()
        sketch_rect = None
        if paths:
            valid_bboxes = [p["rect"] for p in paths if 100 < p["rect"].width < page.rect.width * 0.7]
            if valid_bboxes:
                x0, y0 = min([b.x0 for b in valid_bboxes]), min([b.y0 for b in valid_bboxes])
                x1, y1 = max([b.x1 for b in valid_bboxes]), max([b.y1 for b in valid_bboxes])
                sketch_rect = fitz.Rect(max(0, x0-30), max(0, y0-30), min(page.rect.width, x1+30), min(page.rect.height, y1+30))
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=sketch_rect)
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
                        if sum([parse_val(v) for v in df.iloc[:, c_idx].head(10)]) > 0:
                            s_n = str(df.iloc[0, c_idx]).strip().replace('\n', ' ')
                            specs = {str(df.iloc[d, d_col]).strip(): parse_val(df.iloc[d, c_idx]) for d in range(len(df)) if len(str(df.iloc[d, d_col])) > 3}
                            if specs:
                                if s_n not in all_specs: all_specs[s_n] = {}
                                all_specs[s_n].update(specs)
            
            txt_u = full_txt.upper()
            if any(x in txt_u for x in ["PANT", "QUAN", "SHORT"]): category = "BOTTOM"
            elif any(x in txt_u for x in ["SHIRT", "AO", "TOP"]): category = "TOP"
            vi_summary = translate_insight(full_txt)
            
        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary, "category": category}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Models", f"{count}")
    
    # Hiển thị dung lượng
    used_mb = (count * 0.08)
    st.write(f"💾 **Cloud Storage:** {used_mb:.1f}MB / 1024MB")
    st.progress(min((used_mb / 1024), 1.0))
    
    st.divider()
    new_files = st.file_uploader("Ingest Tech-Packs (Auto-Crop)", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE", use_container_width=True):
        with st.spinner("Synchronizing..."):
            for f in new_files:
                fb = f.read(); h = get_file_hash(fb)
                data = extract_full_techpack(fb)
                if data and data.get('img'):
                    path = f"lib_{h}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], "category": data['category'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
        st.rerun()
    if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode:", ["🔍 AI Search (Audit)", "🔄 Version Comparison"], horizontal=True)

if mode == "🔍 AI Search (Audit)":
    file_audit = st.file_uploader("Upload PDF to Audit:", type="pdf")
    if file_audit:
        with st.status("Isolating sketch and scanning...", expanded=True) as status:
            target = extract_full_techpack(file_audit.read())
            # Phá giới hạn 1000 dòng để quét toàn bộ kho
            all_db = []
            for i in range(0, count, 1000):
                res = supabase.table("ai_data").select("id, vector, file_name, category").range(i, i + 999).execute()
                all_db.extend(res.data)
            df_db = pd.DataFrame(all_db)
            # AI Matching logic...
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            status.update(label="Matching complete!", state="complete")

        cols = st.columns(4)
        cols[0].image(target['img'], caption="TARGET", use_container_width=True)
        for i, (idx, row) in enumerate(top_3.iterrows()):
            det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).single().execute()
            with cols[i+1]:
                st.image(det.data['image_url'], caption=f"Match: {row['sim']:.1%}")
                if st.button(f"SELECT {i+1}", key=f"b_{idx}"):
                    st.session_state['sel_audit'] = {**row.to_dict(), **det.data}

        selected = st.session_state.get('sel_audit')
        if selected:
            all_ex = []
            for sz, t_specs in target['all_specs'].items():
                with st.expander(f"SIZE: {sz}", expanded=True):
                    r_specs = selected['spec_json'].get(sz, {})
                    rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                    for r in rows: r['Diff'] = f"{r['Target'] - r['Ref']:+.3f}"
                    st.table(pd.DataFrame(rows))
                    all_ex.extend([{"Size": sz, **r} for r in rows])
            
            buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
            pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
            st.download_button("📥 DOWNLOAD AUDIT REPORT", buf.getvalue(), f"Audit_{selected['file_name']}.xlsx")

else: # --- MODE: VERSION COMPARISON (CODE MỚI ĐÃ CẬP NHẬT Ở ĐÂY) ---
    st.subheader("🔄 Compare Round A (Repo) vs Round B (New Upload)")
    
    # Lấy toàn bộ danh sách tên file trong kho (vượt 1000 dòng)
    all_names = []
    for i in range(0, count, 1000):
        res_n = supabase.table("ai_data").select("file_name").range(i, i+999).execute()
        all_names.extend([r['file_name'] for r in res_n.data])
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Style A: Select from Repository")
        ver_a_name = st.selectbox("Search Stored SKU:", all_names)
        data_a = supabase.table("ai_data").select("*").eq("file_name", ver_a_name).single().execute().data
        st.image(data_a['image_url'], width=350, caption=f"Stored: {ver_a_name}")
    
    with c2:
        st.info("Style B: Upload New PDF Round")
        file_b = st.file_uploader("Upload New Tech-Pack:", type="pdf", key="file_round_b")
        if file_b:
            data_b = extract_full_techpack(file_b.read())
            if data_b:
                st.image(data_b['img'], width=350, caption="Newly Uploaded Version")

    if file_b and st.button("RUN DIRECT COMPARISON", use_container_width=True):
        st.divider()
        st.success(f"Comparing Stored **{ver_a_name}** with Uploaded **{file_b.name}**")
        all_round_ex = []
        for sz, specs_b in data_b['all_specs'].items():
            with st.expander(f"SIZE: {sz}", expanded=True):
                specs_a = data_a['spec_json'].get(sz, {})
                rows = []
                for p_b, v_b in specs_b.items():
                    m_p = get_close_matches(p_b, list(specs_a.keys()), 1, 0.6)
                    v_a = specs_a.get(m_p[0] if m_p else "", 0)
                    diff = v_b - v_a
                    rows.append({"Point": p_b, "Repo (A)": v_a, "New (B)": v_b, "Diff": f"{diff:+.3f}"})
                    all_round_ex.append({"Size": sz, "Point": p_b, "Stored_A": v_a, "New_B": v_b, "Diff": diff})
                st.table(pd.DataFrame(rows))
        
        buf_r = io.BytesIO(); wr = pd.ExcelWriter(buf_r, engine='xlsxwriter')
        pd.DataFrame(all_round_ex).to_excel(wr, index=False); wr.close()
        st.download_button("📥 DOWNLOAD ROUND COMPARISON (.XLSX)", buf_r.getvalue(), f"Round_Comp_{ver_a_name}.xlsx")
