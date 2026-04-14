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

st.set_page_config(layout="wide", page_title="AI Universal Auditor V113", page_icon="🏢")

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

# ================= 2. MODEL AI & NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {
        "VÁY": ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM"],
        "QUẦN": ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "QUẦN", "DENIM", "JEAN"],
        "ÁO": ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO"]
    }
    scores = {cat: sum(1 for k in keys if k in t) for cat, keys in keywords.items()}
    detected = max(scores, key=scores.get)
    return detected if scores[detected] > 0 else "KHÁC"

def parse_val_universal(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('\n', ' ').strip().lower()
        if any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match_mix = re.search(r'(\d+)\s+(\d+)/(\d+)', txt)
        if match_mix: return float(match_mix.group(1)) + (float(match_mix.group(2)) / float(match_mix.group(3)))
        match_frac = re.search(r'(\d+)/(\d+)', txt)
        if match_frac: return float(match_frac.group(1)) / float(match_frac.group(2))
        match_num = re.search(r'(\d+\.\d+|\d+)', txt)
        if match_num:
            val = float(match_num.group(1))
            return val if val < 500 else 0
        return 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT (FIX LỖI NHẬN NHẦM MÀU) =================
def extract_pdf_v113(file):
    all_specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        full_text = " ".join(full_text_list)
        category = smart_detect_category(full_text, file.name)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.shape[1] < 2: continue
                    
                    n_col, size_cols = -1, {}
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # 1. Tìm cột Description
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["POM DESCRIPTION", "DESCRIPTION", "SPECIFICATION", "POM NAME"]): 
                                n_col = i; break
                        # 2. Tìm Size (Chỉ lấy S,M,L hoặc Số, loại bỏ tên màu dài)
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            cl = v.replace("SIZE", "").replace("-", "").strip()
                            # LOGIC MỚI: Chỉ lấy chuỗi ngắn < 5 ký tự (S, M, L, 32, 10...)
                            if len(cl) > 0 and len(cl) < 6:
                                if cl.isdigit() or cl in ["S", "M", "L", "XL", "XS", "XXL", "2XL", "3XL"]:
                                    size_cols[i] = cl
                        if n_col != -1 and size_cols: break
                    
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                raw_pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                pom_clean = re.sub(r'^[A-Z0-9\.]+\s+', '', raw_pom)
                                val = parse_val_universal(df.iloc[d_idx, s_col])
                                if len(pom_clean) > 4 and val > 0 and "DESCRIPTION" not in pom_clean:
                                    all_specs[s_name][pom_clean] = val
                if all_specs: break 
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except Exception as e: return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", "file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
    except: unique_custs = []

    st.divider()
    cust_input = st.text_input("Nhập tên khách hàng LƯU KHO", value="EVERLANE")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        for f in new_files:
            data = extract_pdf_v113(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({"file_name":f.name,"vector":vec,"spec_json":data['all_specs'],"image_url":supabase.storage.from_(BUCKET).get_public_url(path),"category":data['category'],"customer_name":cust_input.upper()}).execute()
                except Exception as e: st.error(f"Lỗi: {e}")
        st.success("Nạp thành công!"); st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI UNIVERSAL AUDITOR - V113")

c1, c2 = st.columns(2)
with c1: sel_prio = st.selectbox("🎯 Ưu tiên khách hàng từ KHO", ["Tất cả khách hàng", "TP MỚI"] + unique_custs)
with c2: audit_cust = st.text_input("Nhập tên khách cho bản AUDIT này", value="EVERLANE")

file_audit = st.file_uploader("📤 Upload Techpack cần Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang trích xuất dữ liệu...", expanded=True) as status:
        target = extract_pdf_v113(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Đã quét xong: **{list(target['all_specs'].keys())}**")
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
                        col_img, col_table = st.columns([1, 1.8])
                        with col_img:
                            st.image(row['image_url'], use_container_width=True)
                            st.write(f"📌 Khách hàng: **{row['customer_name']}**")
                        with col_table:
                            audit_sizes = sorted(list(target['all_specs'].keys()))
                            sel_s = st.selectbox(f"Chọn Size đối soát mẫu {i+1}:", audit_sizes, key=f"s_{i}")
                            
                            if sel_s:
                                aud_d = target['all_specs'].get(sel_s, {})
                                lib_json = row['spec_json']
                                b_size = sel_s if sel_s in lib_json else list(lib_json.keys())[0]
                                lib_d = lib_json.get(b_size, {})
                                
                                st.write(f"📊 Đối soát: **Audit (S{sel_s})** vs **Gốc (S{b_size})**")
                                res_l = []
                                for k, v_aud in aud_d.items():
                                    v_lib = lib_d.get(k, 0)
                                    diff = round(v_aud - v_lib, 3)
                                    res_l.append({"POM": k, "Mẫu Gốc": v_lib, "Audit": v_aud, "Lệch": diff, "Kết quả": "✅ Khớp" if abs(diff)<0.1 else "❌ Lệch"})
                                st.table(pd.DataFrame(res_l))
            else: st.warning("❌ Không tìm thấy mẫu cùng loại trong kho.")
        else: st.error("❌ Không thể trích xuất thông số. Kiểm tra lại PDF.")
