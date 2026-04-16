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
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

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

# ================= 3. PDF EXTRACTION =================
def extract_pdf_full_logic(file_content):
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        # Chụp ảnh nét cao DPI 2.5 để phóng to soi chi tiết
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_temp = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_temp.save(buf, format="WEBP", quality=80)
        img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages:
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    char_counts = df.apply(lambda x: x.astype(str).str.len().mean())
                    desc_col = char_counts.idxmax()
                    for col_idx in range(len(df.columns)):
                        if col_idx == desc_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, col_idx].head(10)]) > 0:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ')
                            temp_data = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                val = parse_val(df.iloc[d_idx, col_idx])
                                if val > 0 and len(pom) > 2: temp_data[pom] = val
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. DỌN DẸP KHO AN TOÀN (FIXED API ERROR) =================
def safe_deep_clean():
    try:
        files = supabase.storage.from_(BUCKET).list()
        db_res = supabase.table("ai_data").select("id").execute()
        db_ids = [r['id'] for r in db_res.data]
        
        deleted = 0
        for f in files:
            name = f['name']
            f_id = name.replace("lib_", "").split(".")[0]
            # Xóa nếu file không có trong DB hoặc file quá nhỏ (lỗi trắng)
            if f_id not in db_ids or f['metadata'].get('size', 0) < 1000:
                try:
                    supabase.storage.from_(BUCKET).remove([name])
                    supabase.table("ai_data").delete().eq("id", f_id).execute()
                    deleted += 1
                except: continue # Bỏ qua nếu dòng này lỗi, không làm sập app
        return deleted
    except: return 0

# ================= 5. UI SIDEBAR =================
with st.sidebar:
    st.markdown("<h2 style='color: #1E3A8A; margin:0;'>PPJ GROUP</h2>", unsafe_allow_html=True)
    st.title("📂 MASTER REPOSITORY")
    
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Tổng số mẫu", f"{count} SKUs")
    
    if st.button("🧹 DỌN FILE RÁC & LỖI", use_container_width=True):
        num = safe_deep_clean()
        st.success(f"Đã dọn dẹp sạch sẽ {num} mục!")
        time.sleep(1); st.rerun()

    st.divider()
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("ĐỒNG BỘ", use_container_width=True):
        with st.spinner("AI Processing..."):
            for f in new_files:
                fb = f.read(); h = get_file_hash(fb)
                data = extract_pdf_full_logic(fb)
                if data and data.get('img') and data.get('all_specs'):
                    path = f"lib_{h}.webp"
                    # Ép Content-Type để hiện hình
                    supabase.storage.from_(BUCKET).upload(
                        path=path, file=data['img'], 
                        file_options={"content-type": "image/webp", "upsert": "true"}
                    )
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
            st.success("Xong!")
            time.sleep(1); st.rerun()
    
    if st.button("DỌN DANH SÁCH"):
        st.session_state['reset_key'] += 1; st.rerun()

# ================= 6. AUDIT INTERFACE =================
st.title("👔 AI SMART AUDITOR PRO")
file_audit = st.file_uploader("📤 Thả file Audit vào đây", type="pdf")

if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_full_logic(a_bytes)
    if target and target.get("img"):
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # Bộ lọc nhạy đa tầng
            t_name = file_audit.name.upper()
            KEYWORDS = {"CARGO": ["CARGO", "TUI HOP"], "WAIST": ["ELASTIC", "THUN"], "TYPE": ["SKIRT", "VAY", "PANT", "QUAN"]}
            def get_w(row_name):
                row_name = str(row_name).upper(); w = 1.0
                for kw in KEYWORDS["TYPE"]:
                    if (kw in t_name) == (kw in row_name): w += 0.5
                for kw in KEYWORDS["CARGO"] + KEYWORDS["WAIST"]:
                    if (kw in t_name) and (kw in row_name): w += 0.3
                return w

            df_db['score'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in df_db['vector']])).flatten() * df_db['file_name'].apply(get_w)
            top_3 = df_db.sort_values('score', ascending=False).head(3)
            
            st.subheader("🎯 Mẫu tương đồng (Click vào ảnh để phóng to)")
            c_ai = st.columns(4)
            with c_ai[0]: st.image(target['img'], caption="FILE ĐANG AUDIT", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with c_ai[i+1]:
                    st.image(row['image_url'], caption=f"Giống {row['score']:.1%}", use_container_width=True)
                    if st.button(f"CHỌN MẪU {i+1}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['sel'] = row.to_dict()

            best = st.session_state.get('sel')
            if best:
                st.divider()
                st.success(f"🔍 So sánh chi tiết: **{best['file_name']}**")
                if target.get("all_specs"):
                    all_ex = []
                    for sz, t_specs in target['all_specs'].items():
                        with st.expander(f"SIZE: {sz}", expanded=True):
                            r_specs = best['spec_json'].get(sz, {})
                            rows = []; r_poms = list(r_specs.keys())
                            for t_pom, t_val in t_specs.items():
                                m = get_close_matches(t_pom, r_poms, n=1, cutoff=0.6)
                                rv = r_specs.get(m[0], 0) if m else 0
                                rows.append({"Point": t_pom, "Target": t_val, "Ref": rv, "Diff": f"{t_val-rv:+.3f}"})
                                all_ex.append({"Size": sz, "Point": t_pom, "Target": t_val, "Ref": rv})
                            st.table(pd.DataFrame(rows))
                    
                    buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                    pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                    st.download_button("📥 XUẤT BÁO CÁO EXCEL", buf.getvalue(), f"Audit_{best['file_name']}.xlsx", use_container_width=True)
