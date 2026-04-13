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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V81", page_icon="🛡️")

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f0f2f6 !important; }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & PHÂN LOẠI =================
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
    """Chuẩn hóa chuỗi tuyệt đối để so khớp Description"""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper().strip())

# ================= 3. HÀM XỬ LÝ DỮ LIỆU PDF =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_pdf_v81(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += page.get_text()
        doc.close()
        
        category = detect_category(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if any(x in " ".join(row_up) for x in ["DESCRIPTION", "DESC", "POM"]):
                            for i, v in enumerate(row_up):
                                if "DESCRIPTION" in v or "DESC" in v: n_col = i; break
                            if n_col == -1:
                                for i, v in enumerate(row_up):
                                    if "POM" in v: n_col = i; break
                            for i, v in enumerate(row_up):
                                if any(target == v or target in v for target in ["NEW", "SAMPLE", "SPEC", "M", "32"]):
                                    v_col = i; break
                            
                            if n_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                    if len(name) < 3 or any(x in name for x in ["TOL", "REF", "TOTAL"]): continue
                                    val = parse_val(d_row[v_col])
                                    if val > 0: specs[name] = val
                                break
                if specs: break
        
        # ĐIỀU KIỆN LỌC FILE RÁC
        if len(specs) < 3 or img_bytes is None: return None
        return {"specs": specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR (NẠP KHO) =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    res_db = supabase.table("ai_data").select("file_name", "category").execute()
    data_lib = res_db.data if res_db.data else []
    st.info(f"Kho hiện tại: {len(data_lib)} file")

    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới", type="pdf", accept_multiple_files=True)
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        for f in new_files:
            # CHỐNG TRÙNG LẶP
            if any(d['file_name'] == f.name for d in data_lib):
                st.warning(f"⏩ Đã có: {f.name}"); continue
            
            with st.spinner(f"Đang phân tích {f.name}..."):
                data = extract_pdf_v81(f)
                if not data:
                    st.error(f"❌ Loại bỏ {f.name}: Không đủ thông số/ảnh."); continue
                
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category']}).execute()
                st.toast(f"✅ Đã nạp {f.name}")
        st.rerun()

# ================= 5. PHẦN CHÍNH (ĐỐI SOÁT) =================
st.title("🔍 AI SMART AUDITOR - V20.0")

# Upload file kiểm tra
file_audit = st.file_uploader("Kéo thả file Techpack cần kiểm tra vào đây", type="pdf", label_visibility="collapsed")

if file_audit:
    with st.spinner("Đang trích xuất dữ liệu..."):
        target = extract_pdf_v81(file_audit)
    
    if target:
        st.success(f"✨ Loại hàng: **{target['category']}** | {len(target['specs'])} hạng mục thông số.")
        
        # Lọc kho mẫu cùng Category
        db_res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        same_cat_data = db_res.data if db_res.data else []

        if not same_cat_data:
            st.warning(f"⚠️ Trong kho chưa có mẫu nào thuộc loại **{target['category']}** để đối soát.")
        else:
            # --- CHỌN CHẾ ĐỘ TÌM KIẾM ---
            st.divider()
            mode = st.radio("Chế độ so sánh:", ["🤖 Tự động tìm mẫu (AI)", "👆 Chọn mẫu thủ công từ kho"], horizontal=True)
            
            selected_sample = None
            
            if mode == "🤖 Tự động tìm mẫu (AI)":
                img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                
                matches = []
                for item in same_cat_data:
                    v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                    score = float(cosine_similarity(v_test, v_ref)[0][0])
                    matches.append({"item": item, "score": score})
                
                best_match = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                selected_sample = best_match['item']
                st.write(f"✅ AI đã tìm thấy mẫu khớp nhất: **{selected_sample['file_name']}** ({best_match['score']*100:.1f}%)")
            
            else:
                sample_names = [d['file_name'] for d in same_cat_data]
                choice = st.selectbox("Chọn file mẫu từ kho để so sánh:", sample_names)
                selected_sample = next(d for d in same_cat_data if d['file_name'] == choice)

            # --- HIỂN THỊ KẾT QUẢ SO SÁNH ---
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 📄 FILE ĐANG KIỂM")
                st.image(target["img"], use_container_width=True)
                st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))
            
            with c2:
                st.markdown(f"### ✨ MẪU GỐC: {selected_sample['file_name']}")
                st.image(selected_sample['image_url'], use_container_width=True)
                
                # So khớp Description
                ref_specs = selected_sample['spec_json']
                clean_ref_map = {ultra_clean(k): v for k, v in ref_specs.items()}
                
                rows = []
                for k, v in target["specs"].items():
                    k_clean = ultra_clean(k)
                    v_ref = clean_ref_map.get(k_clean, 0) # Lấy giá trị mẫu gốc
                    diff = round(v - v_ref, 3)
                    res = "Khớp" if abs(diff) < 0.125 else "Lệch"
                    rows.append({"Vị trí so sánh": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": res})
                
                df_res = pd.DataFrame(rows)
                st.table(df_res.style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))

            # Nút xuất Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Audit')
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Audit_{selected_sample['file_name']}.xlsx", type="primary")
    else:
        st.error("❌ Không tìm thấy bảng thông số hợp lệ trong PDF.")

# Cuối code: End of session follow-up
# Bạn có cần tôi hỗ trợ viết thêm phần tự động phân tích lý do lệch (ví dụ: nhảy size sai) không?
