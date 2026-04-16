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

# ================= 3. PDF EXTRACTION =================
def extract_pdf_full_logic(file_content):
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        # Chụp ảnh chất lượng cao để Zoom
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_temp = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_temp.save(buf, format="WEBP", quality=85)
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
                            if any(k in s_name.upper() for k in ["TOL", "GRADE", "SPEC"]): continue
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

# ================= 4. UI SIDEBAR & DỌN DẸP KHO =================
with st.sidebar:
    st.markdown("<h2 style='color: #1E3A8A; margin:0;'>PPJ GROUP</h2>", unsafe_allow_html=True)
    st.title("📂 MASTER REPOSITORY")
    
    # Hiển thị SL mẫu và dung lượng
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Tổng số mẫu", f"{count} SKUs")
    st.write(f"💾 **Cloud Storage:** {count * 0.08:.1f}MB / 1GB")
    
    # --- CHỨC NĂNG DỌN DẸP FILE RÁC ---
    if st.button("🧹 QUÉT & DỌN DẸP KHO (XÓA FILE LỖI)", use_container_width=True):
        with st.spinner("Đang tìm và xóa file lỗi..."):
            files_in_storage = supabase.storage.from_(BUCKET).list()
            deleted_count = 0
            for f in files_in_storage:
                # Nếu file quá nhỏ (< 2KB) hoặc không có đuôi chuẩn thường là file rác
                if f['metadata'].get('size', 0) < 2000: 
                    supabase.storage.from_(BUCKET).remove([f['name']])
                    supabase.table("ai_data").delete().eq("id", f['name'].replace("lib_", "").replace(".webp", "").replace(".png", "")).execute()
                    deleted_count += 1
            st.success(f"Đã dọn dẹp {deleted_count} file rác!")
            time.sleep(1); st.rerun()

    st.divider()
    new_files = st.file_uploader("Upload Tech-Pack", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("ĐỒNG BỘ MỚI", use_container_width=True):
        with st.spinner("Đang xử lý AI..."):
            for f in new_files:
                fb = f.read(); h = get_file_hash(fb)
                data = extract_pdf_full_logic(fb)
                if data and data.get('img') and data.get('all_specs'):
                    path = f"lib_{h}.webp"
                    # Ép kiểu image/webp để hiện được hình
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                    supabase.table("ai_data").upsert({
                        "id": h, "file_name": f.name, "vector": get_image_vector(data['img']),
                        "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                    }).execute()
            st.success("Đồng bộ hoàn tất!")
            time.sleep(1); st.rerun()
    
    if st.button("DỌN DẠNH SÁCH UPLOAD"):
        st.session_state['reset_key'] += 1; st.rerun()

# ================= 5. AUDIT INTERFACE =================
st.title("👔 AI SMART AUDITOR PRO")
file_audit = st.file_uploader("📤 Thả file cần Audit vào đây", type="pdf")

if file_audit:
    a_bytes = file_audit.read()
    target = extract_pdf_full_logic(a_bytes)
    if target and target.get("img"):
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            
            # --- BỘ LỌC ĐỘ NHẠY SIÊU CẤP ---
            t_name = file_audit.name.upper()
            KEYWORDS = {
                "CARGO": ["CARGO", "TUI HOP"], "WAIST": ["ELASTIC", "THUN", "LUNG"],
                "TYPE": ["SKIRT", "VAY", "PANT", "QUAN", "SHORT", "TROUSER"]
            }
            def get_weight(row_name):
                row_name = str(row_name).upper(); w = 1.0
                for kw in KEYWORDS["TYPE"]:
                    if (kw in t_name) == (kw in row_name): w += 0.5
                for kw in KEYWORDS["CARGO"] + KEYWORDS["WAIST"]:
                    if (kw in t_name) and (kw in row_name): w += 0.3
                return w

            df_db['weight'] = df_db['file_name'].apply(get_weight)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            df_db['final'] = df_db['sim'] * df_db['weight']
            top_3 = df_db.sort_values('final', ascending=False).head(3)
            
            st.subheader("🎯 Đề xuất mẫu tương đồng (AI)")
            c_ai = st.columns(4)
            with c_ai[0]: st.image(target['img'], caption="FILE ĐANG AUDIT", use_container_width=True)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with c_ai[i+1]:
                    st.image(row['image_url'], caption=f"Giống {row['sim']:.1%}", use_container_width=True)
                    if st.button(f"CHỌN {i+1}", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['sel'] = row.to_dict()

            best = st.session_state.get('sel')
            if best:
                st.divider()
                st.success(f"🔍 So sánh với: **{best['file_name']}**")
                if target.get("all_specs"):
                    all_ex = []
                    for sz, t_specs in target['all_specs'].items():
                        with st.expander(f"SIZE: {sz}", expanded=True):
                            r_specs = best['spec_json'].get(sz, {})
                            rows = []; r_poms = list(r_specs.keys())
                            for t_pom, t_val in t_specs.items():
                                match = get_close_matches(t_pom, r_poms, n=1, cutoff=0.6)
                                rv = r_specs.get(match[0], 0) if match else 0
                                rows.append({"Point": t_pom, "Target": t_val, "Ref": rv, "Diff": f"{t_val-rv:+.3f}"})
                                all_ex.append({"Size": sz, "Point": t_pom, "Target": t_val, "Ref": rv})
                            st.table(pd.DataFrame(rows))
                    
                    buf = io.BytesIO(); wr = pd.ExcelWriter(buf, engine='xlsxwriter')
                    pd.DataFrame(all_ex).to_excel(wr, index=False); wr.close()
                    st.download_button("📥 XUẤT EXCEL", buf.getvalue(), f"Audit_{best['file_name']}.xlsx", use_container_width=True)
