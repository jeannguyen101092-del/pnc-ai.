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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96", page_icon="🛡️")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. MODEL AI & UTILS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "HOODIE", "COAT", "ÁO"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "VÁY"]): return "VÁY/ĐẦM"
    return "KHÁC"

def detect_stage(text):
    """Nhận diện giai đoạn: Production vs Sample/Dev"""
    t = str(text).upper()
    if any(x in t for x in ["PRODUCTION", "PROD", "BULK", "MASS", "SẢN XUẤT"]): return "Production"
    return "Sample/Dev"

def ultra_clean(t):
    if not t: return ""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper().strip())

def parse_val(t):
    try:
        if not t: return 0
        txt = str(t).replace(',', '.').strip().lower().replace('⁄', '/')
        if any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= 3. TRÍCH XUẤT PDF V96 =================
def extract_pdf_v96(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text, file.name)
        stage = detect_stage(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(20).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        # Tìm cột Tên
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "DESC", "POM NAME", "POSITION"]): n_col = i; break
                        # Tìm cột Giá trị (Ưu tiên cột Production nếu có)
                        for i, v in enumerate(row_up):
                            if any(target in v for target in ["PRODUCTION", "PROD", "BULK", "NEW", "SAMPLE", "SPEC", "M", "S", "32"]):
                                if i != n_col: v_col = i; break
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 4 or any(x in name for x in ["TOL", "REF", "REMARK"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "stage": stage}
    except: return None

# ================= 4. SIDEBAR (NẠP KHO) =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    res_db = supabase.table("ai_data").select("file_name", "category", "stage").execute()
    data_lib = res_db.data if res_db.data else []
    st.info(f"Kho hiện tại: {len(data_lib)} file")

    st.divider()
    st.subheader("🚀 NẠP MẪU MỚI")
    new_files = st.file_uploader("Upload Techpack gốc", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        for f in new_files:
            if any(d['file_name'] == f.name for d in data_lib): continue
            with st.spinner(f"Đang nạp {f.name}..."):
                data = extract_pdf_v96(f)
                if data and data['specs']:
                    img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    # Lưu kèm nhãn stage (Production/Sample)
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                        "image_url": url, "category": data['category'], "stage": data['stage']
                    }).execute()
                    st.toast(f"✅ Đã nạp: {f.name} ({data['stage']})")
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. MAIN (SO SÁNH THEO PRODUCTION) =================
st.title("🔍 AI SMART AUDITOR - V96")

# CHỌN CHẾ ĐỘ PHÂN LOẠI
col_mode1, col_mode2 = st.columns(2)
with col_mode1:
    audit_mode = st.radio("Chế độ so sánh:", ["🤖 Tự động (Tìm tất cả)", "🏭 Chỉ tìm trong hàng Production"], horizontal=True)

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra", type="pdf", key="audit_main")

if file_audit:
    with st.spinner("Đang phân tích dữ liệu..."):
        target = extract_pdf_v96(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Phát hiện: **{target['category']}** | Giai đoạn: **{target['stage']}**")
        
        # LOGIC LỌC DATABASE THEO CHẾ ĐỘ
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if "Production" in audit_mode:
            query = query.eq("stage", "Production")
            st.caption(" đang lọc các mẫu thuộc giai đoạn Production...")
        
        db_res = query.execute()
        relevant_data = db_res.data if db_res.data else []

        if not relevant_data:
            st.warning(f"⚠️ Không tìm thấy mẫu nào phù hợp trong kho với tiêu chí đã chọn.")
        else:
            # AI MATCHING TOP 5
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            v_test = model_ai(transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
            
            matches = []
            for item in relevant_data:
                v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref))
                matches.append({"item": item, "score": score})
            
            top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
            
            st.divider()
            sel_match = st.selectbox("🤖 AI gợi ý Top 5 mẫu tương đồng nhất:", top_matches, format_func=lambda x: f"[{x['item']['stage']}] {x['item']['file_name']} ({x['score']*100:.1f}%)")
            selected_sample = sel_match['item']

            # DISPLAY SIDE-BY-SIDE
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📄 ĐANG KIỂM")
                st.image(target["img"], use_container_width=True)
                st.table(pd.DataFrame([{"Vị trí": k, "Số đo": v} for k,v in target["specs"].items()]))
            with c2:
                st.markdown(f"### ✨ MẪU GỐC: {selected_sample['file_name']}")
                st.image(selected_sample['image_url'], use_container_width=True)
                
                clean_ref_map = {ultra_clean(k): v for k, v in selected_sample['spec_json'].items()}
                rows = []
                for k, v in target["specs"].items():
                    v_ref = clean_ref_map.get(ultra_clean(k), 0)
                    diff = round(v - v_ref, 3)
                    res = "Khớp" if abs(diff) < 0.125 else "Lệch"
                    rows.append({"Vị trí": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": res})
                
                df_res = pd.DataFrame(rows)
                st.table(df_res.style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))
            
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer: df_res.to_excel(writer, index=False)
            st.download_button("📥 TẢI BÁO CÁO EXCEL", out.getvalue(), f"Audit_{selected_sample['file_name']}.xlsx", type="primary")
    else:
        st.error("❌ Không thể trích xuất thông số. Hãy kiểm tra PDF có phải dạng Scan ảnh không?")
