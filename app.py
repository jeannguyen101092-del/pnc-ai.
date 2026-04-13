import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V90", page_icon="🛡️")

# Khởi tạo key để reset uploader
if 'up_key' not in st.session_state:
    st.session_state.up_key = 0

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stTable { font-size: 11px !important; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f0f2f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & UTILS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text):
    t = str(text).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "HOODIE"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN"]): return "VÁY/ĐẦM"
    return "KHÁC"

def ultra_clean(t):
    if not t: return ""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper().strip())

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= 3. HÀM TRÍCH XUẤT PDF =================
def extract_pdf(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += (page.get_text() or "")
        doc.close()
        cat = detect_category(full_text)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "DESC"]): n_col = i; break
                        if n_col == -1:
                            for i, v in enumerate(row_up):
                                if "POM" in v: n_col = i; break
                        for i, v in enumerate(row_up):
                            if any(target == v or target in v for target in ["NEW", "SAMPLE", "SPEC", "M", "32", "34"]):
                                if i != n_col: v_col = i; break
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARK"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": cat}
    except: return None

# ================= 4. SIDEBAR (NẠP XONG TỰ XÓA) =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    res_db = supabase.table("ai_data").select("file_name", "category").execute()
    data_lib = res_db.data if res_db.data else []
    st.info(f"Kho hiện tại: {len(data_lib)} file")

    st.divider()
    # Sử dụng up_key để reset khung upload
    new_files = st.file_uploader("Nạp Techpack mới", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        count_success = 0
        for f in new_files:
            if any(d['file_name'] == f.name for d in data_lib):
                st.warning(f"⏩ Đã có: {f.name}"); continue
            with st.spinner(f"Đang nạp {f.name}..."):
                data = extract_pdf(f)
                if data and data['specs']:
                    img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category']}).execute()
                    count_success += 1
        
        if count_success > 0:
            st.success(f"✅ Đã nạp thành công {count_success} file!")
            # TĂNG KEY ĐỂ TỰ XÓA KHUNG UPLOAD
            st.session_state.up_key += 1
            st.rerun()

# ================= 5. MAIN (ĐỐI SOÁT) =================
st.title("🔍 AI SMART AUDITOR - V90")

file_audit = st.file_uploader("📤 Upload file PDF cần đối soát", type="pdf", key="audit_main")

if file_audit:
    with st.spinner("Đang phân tích dữ liệu đối soát..."):
        target = extract_pdf(file_audit)
    
    if target:
        st.success(f"✨ Phát hiện: **{target['category']}** | {len(target['specs'])} vị trí đo.")
        db_all = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        same_cat_data = db_all.data if db_all.data else []

        if not same_cat_data:
            st.warning(f"⚠️ Chưa có mẫu cùng chủng loại trong kho.")
        else:
            st.divider()
            # LOGIC TÌM KIẾM TƯƠNG ĐỒNG TOP 5
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
            
            matches = []
            for item in same_cat_data:
                v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref)[0][0])
                matches.append({"item": item, "score": score})
            
            top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
            
            st.write("### 🤖 Kết quả tương đồng (AI gợi ý Top 5)")
            selected_match = st.selectbox(
                "Chọn mẫu gốc bạn muốn dùng để đối soát:",
                top_matches,
                format_func=lambda x: f"{x['item']['file_name']} (Độ khớp: {x['score']*100:.1f}%)"
            )
            selected_sample = selected_match['item']

            # HIỂN THỊ SONG SONG
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📄 ĐANG KIỂM")
                st.image(target["img"], use_container_width=True)
                st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))
            
            with c2:
                st.markdown(f"### ✨ MẪU GỐC: {selected_sample['file_name']}")
                st.image(selected_sample['image_url'], use_container_width=True)
                
                ref_specs = selected_sample['spec_json']
                clean_ref_map = {ultra_clean(k): v for k, v in ref_specs.items()}
                
                rows = []
                for k, v in target["specs"].items():
                    v_ref = clean_ref_map.get(ultra_clean(k), 0)
                    diff = round(v - v_ref, 3)
                    res = "Khớp" if abs(diff) < 0.125 else "Lệch"
                    rows.append({"Vị trí": k, "Mới": v, "Kho mẫu": v_ref, "Chênh lệch": diff, "Kết quả": res})
                
                df_res = pd.DataFrame(rows)
                st.table(df_res.style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))
            
            # Xuất Excel
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False)
            st.download_button("📥 TẢI BÁO CÁO EXCEL", out.getvalue(), f"Audit_{selected_sample['file_name']}.xlsx", type="primary")
    else:
        st.error("❌ PDF không đủ điều kiện đối soát.")
