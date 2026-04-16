import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state: st.session_state['reset_key'] = 0

# ================= 2. BỘ MÁY AI =================
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
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. TRÍCH XUẤT PDF (BẢN NÂNG CẤP) =================
def extract_pdf_full_logic(file_content):
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_bytes = pix.tobytes("webp")
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages:
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm cột chứa tên điểm đo (thường là cột dài nhất)
                    d_col = df.apply(lambda x: x.astype(str).str.len().mean()).idxmax()
                    
                    for col_idx in range(len(df.columns)):
                        if col_idx == d_col: continue
                        # Kiểm tra xem cột có chứa số liệu không
                        vals = [parse_val(v) for v in df.iloc[:, col_idx]]
                        if sum(vals) > 0:
                            # Lấy tên cột (Size) - thử dòng 0 hoặc dòng 1
                            s_name = str(df.iloc[0, col_idx]).strip() or f"Column_{col_idx}"
                            s_name = s_name.replace('\n', ' ')
                            
                            temp_data = {}
                            for r_idx in range(len(df)):
                                pom = str(df.iloc[r_idx, d_col]).strip().replace('\n', ' ')
                                val = parse_val(df.iloc[r_idx, col_idx])
                                if val > 0 and len(pom) > 2:
                                    temp_data[pom] = val
                            
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. HIỂN THỊ SO SÁNH =================
def render_comparison_logic(target_data, repo_data):
    st.divider()
    st.subheader("📊 GIAI ĐOẠN 1: SO SÁNH HÌNH ẢNH")
    c1, c2 = st.columns(2)
    with c1: st.image(repo_data['image_url'], caption="BẢN GỐC (REPO)", use_container_width=True)
    with c2: st.image(target_data['img'], caption="BẢN MỚI (UPLOAD)", use_container_width=True)

    st.subheader("📊 GIAI ĐOẠN 2: ĐỐI SOÁT THÔNG SỐ")
    t_specs = target_data.get('all_specs', {})
    r_specs = repo_data.get('spec_json', {})

    # Gom tất cả điểm đo về một mối để so khớp tối đa
    t_flat = {k: v for d in t_specs.values() for k, v in d.items()}
    r_flat = {k: v for d in r_specs.values() for k, v in d.items()}
    
    pts = sorted(list(set(t_flat.keys()).intersection(set(r_flat.keys()))))
    
    if pts:
        comp_rows = []
        for p in pts:
            v_b, v_a = t_flat[p], r_flat[p]
            diff = round(v_b - v_a, 3)
            status = "✅ KHỚP" if abs(diff) <= 0.125 else ("❌ LỆCH" if abs(diff) >= 0.5 else "⚠️ NHẸ")
            comp_rows.append({"Điểm đo": p, "Mẫu mới (B)": v_b, "Mẫu gốc (A)": v_a, "Lệch": diff, "Kết quả": status})
        
        df = pd.DataFrame(comp_rows)
        
        # Nút xuất Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Audit')
        st.download_button("📥 Xuất file Excel", output.getvalue(), "Audit_PPJ.xlsx")
        
        st.dataframe(df.style.applymap(lambda v: 'background-color: #ffcccc' if v=="❌ LỆCH" else ('background-color: #fff3cd' if v=="⚠️ NHẸ" else ''), subset=['Kết quả']), use_container_width=True)
    else:
        st.error("❌ Vẫn không tìm thấy điểm đo trùng nhau. Hãy kiểm tra lại định dạng bảng trong PDF.")

# ================= 5. SIDEBAR & MAIN UI =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("SKU trong kho", count)
    st.divider()
    new_files = st.file_uploader("Tải file vào kho", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("ĐỒNG BỘ"):
        for f in new_files:
            fb = f.read(); data = extract_pdf_full_logic(fb)
            if data:
                h = get_file_hash(fb); path = f"lib_{h}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp"})
                supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
        st.rerun()

st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["Tìm kiếm AI", "So sánh Round A vs B"], horizontal=True)

if mode == "Tìm kiếm AI":
    f = st.file_uploader("Upload PDF:", type="pdf")
    if f:
        target = extract_pdf_full_logic(f.read())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            db = pd.DataFrame(res.data)
            db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in db['vector']])).flatten()
            top = db.sort_values('sim', ascending=False).head(3)
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], caption=f"Khớp: {row['sim']:.1%}")
                    if st.button(f"Chọn mẫu {i+1}", key=f"s_{idx}"): st.session_state['selected'] = row.to_dict()
            if 'selected' in st.session_state: render_comparison_logic(target, st.session_state['selected'])
else:
    res = supabase.table("ai_data").select("file_name", "image_url", "spec_json").execute()
    repo_dict = {item['file_name']: item for item in res.data}
    c1, c2 = st.columns(2)
    with c1: sel = st.selectbox("Chọn bản gốc (Repo):", ["-- Chọn --"] + list(repo_dict.keys()))
    with c2: f_new = st.file_uploader("Upload bản mới:", type="pdf")
    if sel != "-- Chọn --" and f_new:
        target_data = extract_pdf_full_logic(f_new.read())
        if target_data: render_comparison_logic(target_data, repo_dict[sel])
