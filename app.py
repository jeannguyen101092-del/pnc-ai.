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
    mapping = {"WASH": "HD Giặt", "FABRIC": "Vải", "STITCH": "May", "LABEL": "Nhãn", "COLOR": "Màu", "POCKET": "Túi", "WAIST": "Lưng"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

# ================= 3. PDF EXTRACTION (SPECIALIZED FOR PPJ) =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary, category = {}, None, "", "UNKNOWN"
    POM_KEYWORDS = ["WAIST", "HIP", "THIGH", "KNEE", "LEG", "INSEAM", "RISE", "LENGTH", "CHEST", "SHOULDER", "SPEC", "MEASURE"]
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_t = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_t.save(buf, format="WEBP", quality=75); img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for p in pdf.pages:
                full_txt += (p.extract_text() or "") + "\n"
                # CHIẾN THUẬT: QUÉT DỰA TRÊN TỌA ĐỘ CHỮ (STREAM MODE)
                tables = p.extract_tables(table_settings={
                    "vertical_strategy": "text", 
                    "horizontal_strategy": "text",
                    "snap_tolerance": 6, # Tăng độ nhạy để gom các số lẻ tẻ
                    "join_tolerance": 6
                })
                
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm cột POM: Cột nào chứa nhiều từ khóa kỹ thuật nhất
                    desc_col = -1
                    max_pom_hits = 0
                    for c_idx in range(len(df.columns)):
                        col_content = " ".join(df.iloc[:, c_idx].astype(str).upper())
                        hits = sum(1 for kw in POM_KEYWORDS if kw in col_content)
                        if hits > max_pom_hits:
                            max_pom_hits = hits
                            desc_col = c_idx
                    
                    if desc_col == -1: continue

                    # Quét các cột khác để tìm cột Size (Có số)
                    for col_idx in range(len(df.columns)):
                        if col_idx == desc_col: continue
                        
                        col_vals = [parse_val(v) for v in df.iloc[:, col_idx]]
                        if sum(1 for v in col_vals if v > 0) >= 2: # Một cột size thường có ít nhất 2 số đo
                            # Lấy tên Size (Thường ở dòng 0 hoặc dòng 1)
                            raw_s_name = str(df.iloc[0, col_idx]).strip() or str(df.iloc[1, col_idx]).strip()
                            s_name = raw_s_name.replace('\n', ' ') if len(raw_s_name) < 15 else f"Size_{col_idx}"
                            
                            if any(k in s_name.upper() for k in ["TOL", "GRADE", "SPEC", "CODE", "POM"]): continue
                            
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom_text = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                val = parse_val(df.iloc[d_idx, col_idx])
                                if val > 0 and len(pom_text) > 3:
                                    # Lọc bỏ rác vải vóc
                                    if not any(x in pom_text.upper() for x in ["COLOR", "FABRIC", "GSM", "TICKET"]):
                                        all_specs[s_name][pom_text] = val
            
            vi_summary = translate_insight(full_txt)
            u_txt = full_txt.upper()
            if any(x in u_txt for x in ["PANT", "QUAN", "SHORT"]): category = "BOTTOM"
            elif any(x in u_txt for x in ["SHIRT", "AO", "TOP"]): category = "TOP"
            
        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary, "category": category}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Synchronized SKU", f"{count}")
    
    # Cloud Storage Display
    st.write(f"💾 **Cloud Storage:** {count * 0.08:.1f}MB / 1024MB")
    st.progress(min((count * 0.08) / 1024, 1.0))
    
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs to Repo", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE", use_container_width=True):
        with st.spinner("Deep Scanning Specs..."):
            for f in new_files:
                fb = f.read(); h = get_file_hash(fb)
                data = extract_full_techpack(fb)
                if data and data.get('all_specs'):
                    path = f"lib_{h}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], 
                        "category": data['category'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
        st.rerun()
    if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode Selection:", ["🔍 AI Search (Audit)", "🔄 Version Control (Repo vs New File)"], horizontal=True)

if mode == "🔍 AI Search (Audit)":
    file_audit = st.file_uploader("Upload Target PDF to Audit:", type="pdf")
    if file_audit:
        target = extract_full_techpack(file_audit.read())
        # (Giữ nguyên logic tìm tương đồng cũ...)
        all_db = []
        for i in range(0, count, 1000):
            res = supabase.table("ai_data").select("id, vector, file_name, category").range(i, i + 999).execute()
            all_db.extend(res.data)
        df_db = pd.DataFrame(all_db)
        t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
        df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
        top_3 = df_db.sort_values('sim', ascending=False).head(3)

        cols = st.columns(4)
        cols.image(target['img'], caption="TARGET", use_container_width=True)
        for i, (idx, row) in enumerate(top_3.iterrows()):
            det = supabase.table("ai_data").select("image_url, spec_json").eq("id", row['id']).single().execute()
            with cols[i+1]:
                st.image(det.data['image_url'], caption=f"Match: {row['sim']:.1%}")
                if st.button(f"SELECT {i+1}", key=f"btn_{idx}"): st.session_state['sel_audit'] = {**row.to_dict(), **det.data}

        selected = st.session_state.get('sel_audit')
        if selected:
            st.divider()
            all_ex = []
            for sz, t_specs in target['all_specs'].items():
                with st.expander(f"SIZE: {sz}", expanded=True):
                    r_sz = get_close_matches(sz, list(selected['spec_json'].keys()), 1, 0.4)
                    r_specs = selected['spec_json'].get(r_sz[0] if r_sz else "", {})
                    rows = []
                    for p, v in t_specs.items():
                        m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                        rv = r_specs.get(m_p[0] if m_p else "", 0)
                        rows.append({"Point": p, "Target": v, "Reference": rv, "Diff": f"{v-rv:+.3f}"})
                        all_ex.append({"Size": sz, **rows[-1]})
                    st.table(pd.DataFrame(rows))
            if all_ex:
                buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                st.download_button("📥 DOWNLOAD AUDIT (.XLSX)", buf.getvalue(), f"Audit.xlsx")

else: # --- MODE: VERSION CONTROL (REPO VS NEW UPLOAD) ---
    st.subheader("🔄 Compare Round A (Repo) vs Round B (New PDF)")
    all_names = []
    for i in range(0, count, 1000):
        res_n = supabase.table("ai_data").select("file_name").range(i, i+999).execute()
        all_names.extend([r['file_name'] for r in res_n.data])
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Step 1: Select SKU from Repository")
        ver_a_name = st.selectbox("Search Model Name:", all_names)
        data_a = supabase.table("ai_data").select("*").eq("file_name", ver_a_name).single().execute().data
        st.image(data_a['image_url'], width=300, caption="Round A (Repo)")
    
    with c2:
        st.info("Step 2: Upload New Round File")
        file_b = st.file_uploader("Drop new PDF here:", type="pdf", key="v_file_b")
        if file_b:
            with st.spinner("Extracting borderless tables..."):
                data_b = extract_full_techpack(file_b.read())
            if data_b:
                st.image(data_b['img'], width=300, caption="Round B (New)")

    if file_b and data_b:
        st.divider()
        st.success(f"📈 **COMPARING:** {ver_a_name} vs {file_b.name}")
        all_round_ex = []
        if not data_b['all_specs']:
            st.error("AI could not detect measurement tables in File B. Please check PDF content.")
        else:
            for sz, specs_b in data_b['all_specs'].items():
                with st.expander(f"SIZE: {sz}", expanded=True):
                    # Khớp Size thông minh
                    r_sz_match = get_close_matches(sz, list(data_a['spec_json'].keys()), 1, 0.4)
                    specs_a = data_a['spec_json'].get(r_sz_match[0] if r_sz_match else "", {})
                    
                    rows = []
                    for p_b, v_b in specs_b.items():
                        m_p = get_close_matches(p_b, list(specs_a.keys()), 1, 0.6)
                        v_a = specs_a.get(m_p[0] if m_p else "", 0)
                        diff = v_b - v_a
                        rows.append({"Measurement Point": p_b, "Stored (A)": v_a, "New (B)": v_b, "Diff": f"{diff:+.3f}"})
                        all_round_ex.append({"Size": sz, **rows[-1]})
                    st.table(pd.DataFrame(rows))
        
        if all_round_ex:
            buf_r = io.BytesIO(); wr = pd.ExcelWriter(buf_r, engine='xlsxwriter')
            pd.DataFrame(all_round_ex).to_excel(wr, index=False); wr.close()
            st.download_button("📥 DOWNLOAD COMPARISON (.XLSX)", buf_r.getvalue(), f"Round_Comparison.xlsx", use_container_width=True)
