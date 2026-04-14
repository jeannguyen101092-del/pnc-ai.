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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V107", page_icon="📏")

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
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP"]
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "QUẦN", "DENIM", "JEAN"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO"]
    s_sk, s_pa, s_sh = sum(1 for k in skirt_keys if k in t), sum(1 for k in pant_keys if k in t), sum(1 for k in shirt_keys if k in t)
    m = max(s_sk, s_pa, s_sh)
    if m == 0: return "KHÁC"
    if m == s_sk: return "VÁY"
    if m == s_pa: return "QUẦN"
    return "ÁO"

def parse_val_safe(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split(); m = float(p[0]); f = p[1].split('/')
            return m + (float(f[0]) / float(f[1]))
        elif '/' in v_str:
            f = v_str.split('/')
            return float(f[0]) / float(f[1])
        return float(v_str)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE (VÉT SẠCH DỮ LIỆU) =================
def extract_pdf_v107(file):
    all_specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        full_text = " ".join(full_text_list)
        category = smart_detect(full_text, file.name)
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    n_col, size_cols = -1, {}
                    # Quét kỹ hàng tiêu đề (quét 30 dòng đầu để chắc chắn)
                    for r_idx, row in df.head(30).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["SPEC", "DESC", "POM", "POS", "POINT"]): n_col = i; break
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            if any(x in v for x in ["TOL", "DIM", "TAGS", "LENGTH"]): continue
                            cl = v.replace("SIZE", "").replace(".0", "").strip()
                            if cl.isdigit() or cl in ["S", "M", "L", "XL", "XS", "XXL", "30", "32", "34"]: size_cols[i] = cl
                        if n_col != -1 and len(size_cols) >= 1: break
                    
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val_safe(df.iloc[d_idx, s_col])
                                if len(pom) > 4 and val > 0 and pom not in ["DESCRIPTION", "SPECIFICATION"]:
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi phân tích file PDF: {e}")
        return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO (BÁO LỖI CHI TIẾT) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", "file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
        existing_files = [x['file_name'] for x in db_res.data]
    except Exception as e:
        st.error(f"Lỗi kết nối Supabase: {e}")
        unique_custs, existing_files = [], []

    st.divider()
    cust_input = st.text_input("Nạp mẫu mới - Nhập tên khách hàng", value="Vineyard Vines")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        count_ok = 0
        for f in new_files:
            if f.name in existing_files:
                st.warning(f"⚠️ {f.name} đã tồn tại trong kho."); continue
            
            data = extract_pdf_v107(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    # Kiểm tra upload ảnh
                    up_img = supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    # Kiểm tra insert database
                    ins_db = supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                        "category": data['category'], "customer_name": cust_input.upper()
                    }).execute()
                    count_ok += 1
                except Exception as e:
                    st.error(f"❌ Lỗi nạp database file {f.name}: {e}")
            else:
                st.error(f"❌ File {f.name} không trích xuất được bảng thông số. Kiểm tra định dạng PDF.")
        
        if count_ok > 0:
            st.success(f"Nạp thành công {count_ok} file!")
            st.session_state.up_key += 1
            st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR V105")
col_sel1, col_sel2 = st.columns(2)
with col_sel1: sel_prio = st.selectbox("🎯 Chọn khách hàng ưu tiên hiển thị", ["Tất cả khách hàng", "TP MỚI"] + unique_custs)
with col_sel2: audit_cust = st.text_input("Tên khách cho Audit hiện tại", "VINEYARD VINES")

file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang quét thông số...", expanded=True) as status:
        target = extract_pdf_v107(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Đã quét xong: **{len(target['all_specs'])} Size** | Loại: **{target['category']}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                df_db['sim_score'] = cosine_similarity(target_vec, np.array([v for v in df_db['vector']])).flatten()
                tar_c = audit_cust.upper().strip()
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 3 if tar_c in str(x).upper() else (2 if sel_prio.upper() in str(x).upper() else 1))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                status.update(label="✅ Đã tìm thấy mẫu khớp!", state="complete", expanded=False)
                
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    st.info(f"📍 Mẫu {i+1}: {row['file_name']} (Khách: {row['customer_name']})")
                    c1, c2 = st.columns(2)
                    with c1: st.image(row['image_url'], use_container_width=True)
                    with c2:
                        common_sizes = sorted(list(set(target['all_specs'].keys()) & set(row['spec_json'].keys())), key=lambda x: int(x) if x.isdigit() else x)
                        sel_s = st.selectbox(f"Chọn Size đối soát mẫu {i+1}:", common_sizes if common_sizes else list(target['all_specs'].keys()), key=f"s_{i}")
                        if sel_s:
                            aud_d = target['all_specs'].get(sel_s, {})
                            lib_s = sel_s if sel_s in row['spec_json'] else list(row['spec_json'].keys())
                            lib_d = row['spec_json'].get(lib_s, {})
                            res_l = [{"POM": k, "Gốc": lib_d.get(k,0), "Audit": v, "Lệch": round(v-lib_d.get(k,0),4)} for k, v in aud_d.items() if k in lib_d]
                            st.table(pd.DataFrame(res_l))
            else: st.warning("Không tìm thấy mẫu cùng loại")
