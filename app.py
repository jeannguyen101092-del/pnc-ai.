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
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. PDF EXTRACTION (DEEP SCAN ALL PAGES) =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary, category = {}, None, "", "UNKNOWN"
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        # Chụp ảnh sketch ở trang đầu tiên
        page_img = doc.load_page(0)
        pix = page_img.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_t = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_t.save(buf, format="WEBP", quality=75); img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            full_txt = ""
            for p in pdf.pages: # QUÉT TẤT CẢ CÁC TRANG TRONG FILE
                full_txt += (p.extract_text() or "") + "\n"
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tự động nhận diện cột Tên thông số (nhiều chữ nhất)
                    char_counts = df.apply(lambda x: x.astype(str).str.len().mean())
                    desc_col = char_counts.idxmax()
                    
                    for col_idx in range(len(df.columns)):
                        if col_idx == desc_col: continue
                        # Kiểm tra xem cột có phải là cột chứa Size không (có ít nhất 3 giá trị số)
                        vals = [parse_val(v) for v in df.iloc[:, col_idx]]
                        if sum(1 for v in vals if v > 0) >= 3:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ')
                            if any(k in s_name.upper() for k in ["TOL", "GRADE", "SPEC", "CODE"]): continue
                            
                            temp_specs = {}
                            for d_idx in range(len(df)):
                                pom_name = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                val = parse_val(df.iloc[d_idx, col_idx])
                                if val > 0 and len(pom_name) > 3:
                                    temp_specs[pom_name] = val
                            
                            if temp_specs:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_specs) # Gộp thông số từ các bảng khác nhau
            
            txt_u = full_txt.upper()
            if any(x in txt_u for x in ["PANT", "QUAN", "SHORT"]): category = "BOTTOM"
            elif any(x in txt_u for x in ["SHIRT", "AO", "TOP"]): category = "TOP"
            
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Total Models", f"{count}")
    st.write(f"💾 **Cloud Storage:** {count * 0.08:.1f}MB / 1024MB")
    st.progress(min((used_mb := count * 0.08) / 1024, 1.0))
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("SYNCHRONIZE", use_container_width=True):
        with st.spinner("Saving to Repository..."):
            for f in new_files:
                fb = f.read(); h = get_file_hash(fb)
                data = extract_full_techpack(fb)
                if data and data.get('img') and data.get('all_specs'):
                    path = f"lib_{h}.webp"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "category": data['category'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
        st.rerun()
    if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode Selection:", ["🔍 AI Search (Audit)", "🔄 Version Control (Repo vs New File)"], horizontal=True)

if mode == "🔍 AI Search (Audit)":
    file_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if file_audit:
        with st.status("Deep Scanning all pages...", expanded=True) as status:
            target = extract_full_techpack(file_audit.read())
            all_db = []
            for i in range(0, count, 1000):
                res = supabase.table("ai_data").select("id, vector, file_name, category").range(i, i + 999).execute()
                all_db.extend(res.data)
            df_db = pd.DataFrame(all_db)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            status.update(label="Scanning Complete!", state="complete")

        cols = st.columns(4)
        cols[0].image(target['img'], caption="TARGET (SKETCH)", use_container_width=True)
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
                    rows = []
                    for p, v in t_specs.items():
                        m_p = get_close_matches(p, list(r_specs.keys()), 1, 0.6)
                        rv = r_specs.get(m_p[0] if m_p else "", 0)
                        rows.append({"Point": p, "Target": v, "Ref": rv, "Diff": f"{v-rv:+.3f}"})
                        all_ex.append({"Size": sz, "Point": p, "Target": v, "Ref": rv, "Diff": v-rv})
                    st.table(pd.DataFrame(rows))
            
            buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
            pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
            st.download_button("📥 DOWNLOAD AUDIT REPORT", buf.getvalue(), f"Audit_{selected['file_name']}.xlsx", use_container_width=True)

else: # --- VERSION CONTROL MODE (REPO VS NEW UPLOAD) ---
    st.subheader("🔄 Compare Round A (From Repository) vs Round B (New PDF)")
    
    # Lấy toàn bộ danh sách file từ kho (vượt giới hạn 1000)
    all_names = []
    for i in range(0, count, 1000):
        res_n = supabase.table("ai_data").select("file_name").range(i, i+999).execute()
        all_names.extend([r['file_name'] for r in res_n.data])
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Step 1: Select SKU from Repository")
        ver_a_name = st.selectbox("Search Model Name:", all_names)
        data_a = supabase.table("ai_data").select("*").eq("file_name", ver_a_name).single().execute().data
        st.image(data_a['image_url'], width=300, caption="Stored Version")
    
    with c2:
        st.info("Step 2: Upload New Tech-Pack File")
        file_b = st.file_uploader("Drop new Round file here:", type="pdf", key="v_file_b")
        if file_b:
            with st.spinner("Scanning all pages of new file..."):
                data_b = extract_full_techpack(file_b.read())
            if data_b:
                st.image(data_b['img'], width=300, caption="New Version")

    # TỰ ĐỘNG HIỂN THỊ KẾT QUẢ KHI CÓ ĐỦ 2 BÊN
    if file_b and data_b:
        st.divider()
        st.success(f"📈 **COMPARING:** {ver_a_name} (Repo) vs {file_b.name} (New)")
        
        all_round_ex = []
        # Ưu tiên lấy theo Size của file mới vừa upload
        for sz, specs_b in data_b['all_specs'].items():
            with st.expander(f"SIZE: {sz}", expanded=True):
                # Tìm size tương ứng trong kho
                specs_a = data_a['spec_json'].get(sz, {})
                rows = []
                for p_b, v_b in specs_b.items():
                    # Khớp tên vị trí đo mờ (Fuzzy matching)
                    match_p = get_close_matches(p_b, list(specs_a.keys()), 1, 0.6)
                    v_a = specs_a.get(match_p[0] if match_p else "", 0)
                    diff = v_b - v_a
                    rows.append({"Point": p_b, "Stored (A)": v_a, "New (B)": v_b, "Diff": f"{diff:+.3f}"})
                    all_round_ex.append({"Size": sz, "Point": p_b, "Stored_A": v_a, "New_B": v_b, "Diff": diff})
                
                if rows: st.table(pd.DataFrame(rows))
                else: st.warning(f"No matching measurement points found for Size {sz}")
        
        if all_round_ex:
            buf_r = io.BytesIO(); wr = pd.ExcelWriter(buf_r, engine='xlsxwriter')
            pd.DataFrame(all_round_ex).to_excel(wr, index=False); wr.close()
            st.download_button("📥 DOWNLOAD COMPARISON (.XLSX)", buf_r.getvalue(), f"Comparison_{ver_a_name}.xlsx", use_container_width=True)
