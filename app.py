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
if 'sel' not in st.session_state: st.session_state['sel'] = None

def display_logo(width=200):
    if os.path.exists("logo.png"): st.image("logo.png", width=width)
    else: st.markdown(f"<h1 style='color: #1E3A8A; margin:0;'>PPJ GROUP</h1>", unsafe_allow_html=True)

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
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. PDF EXTRACTION (QUÉT TRỌN BỘ THÔNG SỐ) =================
def extract_pdf_multi_size(file_content):
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    char_counts = df.apply(lambda x: x.astype(str).str.len().mean())
                    desc_col = char_counts.idxmax()
                    for col_idx in range(len(df.columns)):
                        if col_idx == desc_col: continue
                        sample_vals = [parse_val(v) for v in df.iloc[:, col_idx].head(20)]
                        if sum(sample_vals) > 0:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ') or f"Size_{col_idx}"
                            if any(kw in s_name.upper() for kw in ["TOL", "GRADE", "SPEC", "CODE"]): continue
                            temp_data = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                val = parse_val(df.iloc[d_idx, col_idx])
                                if val > 0 and len(pom) > 2:
                                    temp_data[pom] = val
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. UI MASTER REPOSITORY =================
with st.sidebar:
    display_logo(width=220)
    st.markdown("---")
    st.title("📂 MASTER REPOSITORY")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Synchronized SKUs", f"{count} Models")
    
    used_mb = (count * 0.15)
    st.write(f"💾 **Cloud Storage:** {used_mb:.1f}MB / 1GB")
    st.progress(min((used_mb / 1024), 1.0))

    st.divider()
    st.subheader("📥 Data Ingestion")
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        if new_files and st.button("SYNCHRONIZE", use_container_width=True):
            with st.spinner("AI Processing..."):
                new_count = 0
                for f in new_files:
                    fb = f.read(); h = get_file_hash(fb)
                    data = extract_pdf_multi_size(fb)
                    # BẮT BUỘC CÓ THÔNG SỐ MỚI CHO UP
                    if data and data.get('img') and data.get('all_specs') and len(data['all_specs']) > 0:
                        path = f"lib_{h}.png"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                        supabase.table("ai_data").upsert({
                            "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                            "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                        }).execute()
                        new_count += 1
                    else:
                        st.sidebar.error(f"❌ {f.name}: Không có thông số.")
                if new_count > 0: st.sidebar.success(f"✅ Đã cập nhật {new_count} mẫu!")
            time.sleep(2); st.rerun()
    with col_up2:
        if st.button("CLEAR FILES", use_container_width=True):
            st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. AUDIT INTERFACE =================
h_col1, h_col2 = st.columns(2)
with h_col1: display_logo(width=120)
with h_col2: st.title("AI SMART AUDITOR PRO")

st.markdown("---")
file_audit = st.file_uploader("📤 Drag & Drop Tech-Pack for Auditing", type="pdf")

if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_multi_size(a_bytes)
    
    if target and target.get("img"):
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # TÌM KIẾM THỦ CÔNG
            st.subheader("🔍 Tìm kiếm mã hàng trong kho")
            search_query = st.text_input("Nhập mã hàng/tên file:", placeholder="Ví dụ: 5176...")
            if search_query:
                df_src = df_db[df_db['file_name'].str.contains(search_query, case=False, na=False)]
                if not df_src.empty:
                    s_cols = st.columns(min(len(df_src), 4))
                    for i, (idx, s_row) in enumerate(df_src.head(4).iterrows()):
                        with s_cols[i]:
                            st.image(s_row['image_url'], width=100)
                            if st.button(f"CHỌN {s_row['file_name'][:10]}...", key=f"src_{idx}"):
                                st.session_state['sel'] = s_row.to_dict()

            st.divider()
            if st.button("🚀 TỰ ĐỘNG TÌM KIẾM MẪU TƯƠNG ĐỒNG (AI)", use_container_width=True):
                st.session_state['sel'] = None

            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            
            if st.session_state['sel'] is None:
                st.subheader("🎯 Đề xuất mẫu tương đồng (AI)")
                cols_ai = st.columns(4) # SỬA LỖI TẠI ĐÂY
                with cols_ai[0]: 
                    st.image(target['img'], caption="SOURCE SKETCH", use_container_width=True)
                for i, (idx, row) in enumerate(top_3.iterrows()):
                    with cols_ai[i+1]:
                        st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}", use_container_width=True)
                        if st.button(f"SELECT MODEL {i+1}", key=f"btn_{idx}", use_container_width=True):
                            st.session_state['sel'] = row.to_dict()

            best = st.session_state.get('sel')
            if best:
                st.success(f"**ĐỐI ỨNG VỚI MẪU:** {best['file_name']}")
                if target.get("all_specs"):
                    st.subheader("📋 Measurement Comparison")
                    all_export = []
                    for sz, t_specs in target['all_specs'].items():
                        with st.expander(f"SIZE: {sz}", expanded=True):
                            r_specs = best['spec_json'].get(sz, {})
                            rows = []; r_poms = list(r_specs.keys())
                            for t_pom, t_val in t_specs.items():
                                matches = get_close_matches(t_pom, r_poms, n=1, cutoff=0.6)
                                rv = r_specs.get(matches[0], 0) if matches else 0
                                rows.append({"Point": t_pom, "Target": t_val, "Ref": rv, "Diff": f"{t_val-rv:+.3f}"})
                                all_export.append({"Size": sz, "Point": t_pom, "Target": t_val, "Ref": rv, "Diff": t_val-rv})
                            st.table(pd.DataFrame(rows))
                    
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as wr:
                        pd.DataFrame(all_export).to_excel(wr, index=False)
                    st.download_button("📥 DOWNLOAD REPORT", buf.getvalue(), f"Audit_{best['file_name']}.xlsx", use_container_width=True)
