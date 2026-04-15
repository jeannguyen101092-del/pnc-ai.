import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Universal Auditor V114", page_icon="🏢")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f8f9fa !important; color: #333 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. CÔNG CỤ NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def smart_detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()

    # --- CLEAN TEXT ---
    t = re.sub(r"[^A-Z0-9 ]", " ", t)

    # --- KEYWORDS MỞ RỘNG ---
    keywords = {
        "VÁY": [
            "SKIRT", "DRESS", "GOWN", "FROCK",
            "MINI SKIRT", "MAXI SKIRT", "ONE PIECE", "ONEPIECE",
            "VAY", "DAM"
        ],
        "QUẦN": [
            "PANT", "PANTS", "JEAN", "JEANS", "TROUSER", "TROUSERS",
            "SHORT", "SHORTS", "JOGGER", "LEGGING",
            "INSEAM", "OUTSEAM", "THIGH", "KNEE", "LEG OPENING",
            "WAIST", "HIP", "RISE",
            "QUAN"
        ],
        "ÁO": [
            "SHIRT", "TSHIRT", "T-SHIRT", "TEE",
            "JACKET", "COAT", "HOODIE", "SWEATER",
            "BLAZER", "POLO", "TOP",
            "CHEST", "BUST", "SLEEVE", "SHOULDER",
            "LENGTH", "CUFF", "COLLAR",
            "AO"
        ]
    }

    # --- WEIGHT (ưu tiên từ khóa quan trọng) ---
    strong_keywords = {
        "QUẦN": ["INSEAM", "OUTSEAM", "THIGH", "RISE"],
        "ÁO": ["CHEST", "BUST", "SLEEVE"],
        "VÁY": ["DRESS", "SKIRT"]
    }

    scores = {cat: 0 for cat in keywords}

    # --- TÍNH ĐIỂM ---
    for cat, keys in keywords.items():
        for k in keys:
            if k in t:
                scores[cat] += 1

    # --- BONUS ĐIỂM KEY QUAN TRỌNG ---
    for cat, keys in strong_keywords.items():
        for k in keys:
            if k in t:
                scores[cat] += 2

    # --- LOGIC ƯU TIÊN NGỮ CẢNH ---
    # nếu có nhiều từ đo size đặc trưng → override
    if any(x in t for x in ["INSEAM", "THIGH", "LEG OPENING"]):
        return "QUẦN"
    if any(x in t for x in ["CHEST", "BUST", "SLEEVE"]):
        return "ÁO"

    # --- CHỌN KẾT QUẢ ---
    detected = max(scores, key=scores.get)

    # --- FALLBACK THÔNG MINH ---
    if scores[detected] == 0:
        if "SET" in t or "SUIT" in t:
            return "BỘ"
        return "KHÁC"

def parse_val_universal(t):
    try:
        if not t: return 0
        txt = str(t).replace(',', '.').replace('\n', ' ').strip().lower()
        match = re.search(r'(\d+)\s+(\d+)/(\d+)', txt) # 14 1/2
        if match: return float(match.group(1)) + (float(match.group(2)) / float(match.group(3)))
        match = re.search(r'(\d+)/(\d+)', txt) # 1/2
        if match: return float(match.group(1)) / float(match.group(2))
        match = re.search(r'(\d+\.\d+|\d+)', txt)
        return float(match.group(1)) if match else 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE (THE PRO SCANNER) =================
import re, io, fitz, pdfplumber, pandas as pd

# ================== 1. NHẬN DIỆN BRAND ==================
def detect_brand(text):
    t = text.upper()
    brands = {
        "LEVI": ["LEVI", "LS&CO"],
        "NIKE": ["NIKE"],
        "ADIDAS": ["ADIDAS"],
        "UNIQLO": ["UNIQLO"],
        "H&M": ["H&M", "HM"],
        "ZARA": ["ZARA"],
    }
    for b, keys in brands.items():
        if any(k in t for k in keys):
            return b
    return "GENERIC"


# ================== 2. CHUẨN HOÁ POM ==================
def normalize_pom(pom):
    p = pom.upper().strip()

    mapping = {
        "WAIST": ["WAIST", "WAISTBAND"],
        "HIP": ["HIP", "SEAT"],
        "THIGH": ["THIGH"],
        "INSEAM": ["INSEAM"],
        "OUTSEAM": ["OUTSEAM"],
        "LEG OPENING": ["LEG OPENING", "LEG OPEN"],
        "FRONT RISE": ["FRONT RISE"],
        "BACK RISE": ["BACK RISE"],
        "CHEST": ["CHEST", "BUST"],
        "LENGTH": ["LENGTH", "BODY LENGTH"],
        "SLEEVE": ["SLEEVE", "SLEEVE LENGTH"],
        "SHOULDER": ["SHOULDER"],
    }

    for std, keys in mapping.items():
        if any(k in p for k in keys):
            return std
    return p


# ================== 3. PARSE SIZE ==================
def clean_size(val):
    v = str(val).upper().strip()
    v = re.sub(r"SIZE|-", "", v).strip()
    if len(v) <= 6:
        return v
    return None


# ================== 4. TÌM HEADER THÔNG MINH ==================
def detect_structure(df):
    n_col = -1
    size_cols = {}

    for r_idx, row in df.iterrows():
        row_up = [str(c).upper().strip() for c in row]

        # tìm cột POM
        for i, v in enumerate(row_up):
            if any(x in v for x in ["POM", "DESCRIPTION", "SPEC", "MEASUREMENT"]):
                n_col = i
                break

        # tìm size
        for i, v in enumerate(row_up):
            if i == n_col:
                continue
            s = clean_size(v)
            if s:
                size_cols[i] = s

        if n_col != -1 and size_cols:
            return n_col, size_cols

    return -1, {}


# ================== 5. CORE EXTRACT ==================
def extract_pdf_v2(file):
    all_specs, img_bytes, full_text_list = {}, None, []

    try:
        file.seek(0)
        pdf_content = file.read()

        # --- READ PDF ---
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc:
            full_text_list.append(str(page.get_text() or ""))
        doc.close()

        full_text = " ".join(full_text_list)

        # --- DETECT ---
        category = smart_detect_category(full_text, file.name)
        brand = detect_brand(full_text)

        # --- PARSE TABLE ---
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")

                    n_col, size_cols = detect_structure(df)
                    if n_col == -1:
                        continue

                    for s_col, s_name in size_cols.items():
                        if s_name not in all_specs:
                            all_specs[s_name] = {}

                        for r in range(len(df)):
                            raw_pom = str(df.iloc[r, n_col]).strip()
                            raw_pom = re.sub(r'^[A-Z0-9\.]+\s+', '', raw_pom)

                            pom = normalize_pom(raw_pom)
                            val = parse_val_universal(df.iloc[r, s_col])

                            if len(pom) > 2 and val > 0:
                                all_specs[s_name][pom] = val

                if all_specs:
                    break

        return {
            "all_specs": all_specs,
            "img": img_bytes,
            "category": category,
            "brand": brand
        }

    except Exception as e:
        print("ERROR:", e)
        return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", "file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
    except: unique_custs = []

    st.divider()
    cust_input = st.text_input("Nhập tên khách hàng LƯU KHO", value="EXPRESS")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        for f in new_files:
            data = extract_pdf_v114(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({"file_name":f.name,"vector":vec,"spec_json":data['all_specs'],"image_url":supabase.storage.from_(BUCKET).get_public_url(path),"category":data['category'],"customer_name":cust_input.upper()}).execute()
                except Exception as e: st.error(f"Lỗi: {e}")
        st.success("Nạp thành công!"); st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH (CROSS-BRAND MATCHING) =================
st.title("🔍 AI UNIVERSAL AUDITOR - V114")

col1, col2 = st.columns(2)
with col1: sel_prio = st.selectbox("🎯 Ưu tiên khách hàng từ KHO", ["Tất cả khách hàng", "TP MỚI"] + unique_custs)
with col2: audit_cust = st.text_input("Nhập tên khách cho bản AUDIT này", value="EXPRESS")

file_audit = st.file_uploader("📤 Upload Techpack cần Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang phân tích dữ liệu đa thương hiệu...", expanded=True) as status:
        target = extract_pdf_v114(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Đã quét xong Audit: **{list(target['all_specs'].keys())}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                db_vecs = np.array([v for v in df_db['vector']])
                df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
                
                tar_c = audit_cust.upper().strip()
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 3 if tar_c in str(x).upper() else (2 if sel_prio.upper() in str(x).upper() else 1))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                status.update(label="✅ Đã tìm thấy mẫu tương đồng!", state="complete", expanded=False)

                tabs = st.tabs([f"Top {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    with tabs[i]:
                        c_img, c_table = st.columns([1, 1.8])
                        with c_img:
                            st.image(row['image_url'], use_container_width=True)
                            st.write(f"📌 Khách hàng gốc: **{row['customer_name']}**")
                        with c_table:
                            audit_sizes = sorted(list(target['all_specs'].keys()))
                            sel_s = st.selectbox(f"Chọn Size bản Audit (Mã {i+1}):", audit_sizes, key=f"s_{i}")
                            
                            if sel_s:
                                aud_d = target['all_specs'].get(sel_s, {})
                                lib_json = row['spec_json']
                                
                                # FIX LỖI "EMPTY": Nếu không trùng tên size, lấy size đầu tiên của mẫu gốc
                                b_size = sel_s if sel_s in lib_json else list(lib_json.keys())[0]
                                lib_d = lib_json.get(b_size, {})
                                
                                st.write(f"📊 Đối soát: **Audit (S{sel_s})** vs **Gốc (S{b_size})**")
                                
                                # LOGIC KHỚP Tên Vị Trí (POM) THÔNG MINH
                                res_l = []
                                for k_aud, v_aud in aud_d.items():
                                    # Tìm POM trong mẫu gốc có độ giống cao nhất hoặc chứa từ khóa
                                    v_lib = 0
                                    match_key = "Không tìm thấy"
                                    for k_lib in lib_d.keys():
                                        if k_aud in k_lib or k_lib in k_aud: # Khớp từ khóa (VD: WAIST trong WAIST CIRCUM)
                                            v_lib = lib_d[k_lib]
                                            match_key = k_lib
                                            break
                                    
                                    diff = round(v_aud - v_lib, 3) if v_lib > 0 else 0
                                    res_l.append({
                                        "Audit POM": k_aud, 
                                        "Mẫu Gốc POM": match_key,
                                        "Mẫu Gốc Val": v_lib if v_lib > 0 else "-", 
                                        "Audit Val": v_aud, 
                                        "Chênh lệch": diff if v_lib > 0 else "-",
                                        "Kết quả": "✅ Khớp" if abs(diff)<0.1 and v_lib>0 else "❌"
                                    })
                                st.table(pd.DataFrame(res_l))
            else: st.warning("❌ Không tìm thấy mẫu cùng loại hàng trong kho.")
