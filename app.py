import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state: st.session_state['reset_key'] = 0
if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None

# ================= 2. AI ENGINE =================
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

def translate_insight(text):
    mapping = {"WASH": "HD Giặt", "FABRIC": "Vải", "STITCH": "Quy cách May", "LABEL": "Nhãn", "COLOR": "Màu", "POCKET": "Túi", "WAIST": "Lưng/Cạp"}
    notes = [f"**[{v}]**: {l.strip()}" for l in text.split('\n') for k, v in mapping.items() if k in l.upper() and len(l.strip()) > 5]
    return "\n\n".join(list(set(notes))[:10])

# ================= 3. PDF EXTRACTION =================
def extract_full_techpack(file_content):
    all_specs, img_bytes, vi_summary = {}, None, ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0)) 
        img_t = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_t.save(buf, format="WEBP", quality=70); img_bytes = buf.getvalue()
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
                        if sum([parse_val(v) for v in df.iloc[:, c_idx].head(15)]) > 0:
                            s_n = str(df.iloc[0, c_idx]).strip().replace('\n', ' ')
                            if any(k in s_n.upper() for k in ["TOL", "GRADE", "SPEC", "CODE"]): continue
                            specs = {str(df.iloc[d, d_col]).strip(): parse_val(df.iloc[d, c_idx]) for d in range(len(df)) if len(str(df.iloc[d, d_col])) > 3}
                            if specs:
                                if s_n not in all_specs: all_specs[s_n] = {}
                                all_specs[s_n].update(specs)
            vi_summary = translate_insight(full_txt)
        return {"all_specs": all_specs, "img": img_bytes, "summary_vi": vi_summary}
    except: return None

# ================= 4. UI SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Repository SKUs", f"{count} Models")
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
                        supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "summary_vi": data['summary_vi'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
            st.rerun()
    with c2:
        if st.button("CLEAR LIST"): st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Mode:", ["🔍 AI Search (Audit)", "🔄 Version Comparison"], horizontal=True)

if mode == "🔍 AI Search (Audit)":
    file_audit = st.file_uploader("Upload PDF to find match:", type="pdf")
    if file_audit:
        target = extract_full_techpack(file_audit.read())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            df_db = pd.DataFrame(res.data)
            
            # --- AI Matching ---
            t_name = file_audit.name.upper()
            def hybrid_filter(row_name):
                row_name = str(row_name).upper()
                is_t_pant = any(x in t_name for x in ["PANT", "QUAN", "SHORT", "TROUSER"])
                is_r_pant = any(x in row_name for x in ["PANT", "QUAN", "SHORT", "TROUSER"])
                if is_t_pant != is_r_pant: return 0.0
                return 1.3 if ("CARGO" in t_name) == ("CARGO" in row_name) else 1.0

            df_db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in df_db['vector']])).flatten()
            df_db['final'] = df_db['sim'] * df_db['file_name'].apply(hybrid_filter)
            top_3 = df_db[df_db['final'] > 0].sort_values('final', ascending=False).head(3)
            
            st.subheader("🎯 Top Matches")
            cols = st.columns(4)
            cols[0].image(target['img'], caption="TARGET FILE", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i+1]:
                    st.image(row['image_url'], caption=f"Match: {row['sim']:.1%}")
                    if st.button(f"COMPARE WITH THIS", key=f"btn_{idx}"): st.session_state['sel_audit'] = row.to_dict()

            # ================= 6. COMPARISON SECTION (HÌNH ẢNH TRƯỚC) =================
            selected = st.session_state.get('sel_audit')
            if selected:
                st.divider()
                st.subheader(f"🔄 CHI TIẾT ĐỐI SOÁT: {selected['file_name']}")
                
                # --- PHẦN 1: HÌNH ẢNH ---
                st.write("#### 1. So sánh hình dáng & Quy cách")
                v_c1, v_c2 = st.columns(2)
                v_c1.image(target['img'], caption="BẢN MỚI (TARGET)", use_container_width=True)
                v_c2.image(selected['image_url'], caption=f"BẢN GỐC (REPO) - Độ khớp: {selected['sim']:.1%}", use_container_width=True)
                
                with st.expander("📝 Ghi chú kỹ thuật song song", expanded=True):
                    n_c1, n_c2 = st.columns(2)
                    n_c1.info(target['summary_vi'] if target['summary_vi'] else "Không có ghi chú.")
                    n_c2.success(selected['summary_vi'] if selected['summary_vi'] else "Không có ghi chú.")

                # --- PHẦN 2: THÔNG SỐ ---
                st.write("#### 2. So sánh thông số (Specs)")
                target_tables = target['all_specs']
                ref_tables = selected['spec_json']
                common_tables = list(set(target_tables.keys()).intersection(set(ref_tables.keys())))
                
                if common_tables:
                    t_key = st.selectbox("Chọn bảng size đối soát:", common_tables)
                    t_data, r_data = target_tables[t_key], ref_tables[t_key]
                    
                    all_pts = sorted(list(set(t_data.keys()).intersection(set(r_data.keys()))))
                    if all_pts:
                        rows = []
                        for pt in all_pts:
                            v1, v2 = t_data[pt], r_data[pt]
                            diff = round(v1 - v2, 3)
                            status = "✅ Khớp" if abs(diff) <= 0.125 else ("❌ Lệch" if abs(diff) >= 0.5 else "⚠️ Nhẹ")
                            rows.append({"Điểm đo": pt, "Bản Mới": v1, "Bản Gốc": v2, "Chênh lệch": diff, "Kết quả": status})
                        
                        df_comp = pd.DataFrame(rows)
                        def style_col(v):
                            if v == "❌ Lệch": return 'background-color: #ffcccc'
                            if v == "⚠️ Nhẹ": return 'background-color: #fff3cd'
                            return ''
                        st.dataframe(df_comp.style.applymap(style_col, subset=['Kết quả']), use_container_width=True)
                else:
                    st.warning("Không tìm thấy bảng thông số tương đồng để so sánh tự động.")

else:
    st.info("Chế độ so sánh thủ công giữa 2 phiên bản đang được cập nhật.")
