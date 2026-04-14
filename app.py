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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V106", page_icon="📏")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# CSS làm đẹp giao diện giống ảnh
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f8f9fa !important; color: #333 !important; }
    .stAlert { padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP", "WAIST TO HEM"]
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "TROUSER", "QUẦN", "FLY-STITCH", "CROTCH"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET"]
    s_sk, s_pa, s_sh = sum(1 for k in skirt_keys if k in t), sum(1 for k in pant_keys if k in t), sum(1 for k in shirt_keys if k in t)
    m = max(s_sk, s_pa, s_sh)
    if m == 0: return "KHÁC"
    if m == s_sk: return "VÁY"
    if m == s_pa: return "QUẦN"
    return "ÁO"

def parse_val_safe(t):
    """Xử lý phân số Denim (VD: 9 3/4)"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split(); main_v = float(p[0])
            frac = p[1].split('/'); return main_v + (float(frac[0]) / float(frac[1]))
        elif '/' in v_str:
            frac = v_str.split('/'); return float(frac[0]) / float(frac[1])
        return float(v_str)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE (DENIM MASTER OPTIMIZED) =================
def extract_pdf_pro(file):
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
                    # Dò hàng tiêu đề để tìm cột Description và các cột Size
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # 1. Tìm cột POM
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "SPEC", "POM", "POS"]): 
                                n_col = i; break
                        # 2. Tìm cột Size (Bỏ qua DIM, TOL, LENGTH)
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            if any(x in v for x in ["TOL", "DIM", "TAGS", "LENGTH", "REF", "+", "-"]): continue
                            clean_v = v.replace("SIZE", "").strip()
                            if clean_v.isdigit() or clean_v in ["S", "M", "L", "XL", "XXL"]: size_cols[i] = clean_v
                        if n_col != -1 and len(size_cols) >= 1: break
                    
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val_safe(df.iloc[d_idx, s_col])
                                if len(pom) > 4 and val > 0 and pom != "DESCRIPTION":
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR - KHO THIẾT KẾ MẪU =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", "file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
        existing_files = [x['file_name'] for x in db_res.data]
    except: unique_custs, existing_files = [], []

    st.divider()
    cust_input = st.text_input("Nạp mẫu mới - Nhập tên khách hàng", value="Vineyard Vines")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        for f in new_files:
            if f.name in existing_files:
                st.warning(f"⚠️ {f.name} đã tồn tại."); continue
            data = extract_pdf_pro(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                        "category": data['category'], "customer_name": cust_input.upper()
                    }).execute()
                except Exception as e: st.error(f"Lỗi DB: {e}")
        st.success("Nạp thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR V105")

col_h1, col_h2 = st.columns(2)
with col_h1: sel_prio = st.selectbox("🎯 Chọn khách hàng ưu tiên hiển thị", ["Tất cả khách hàng", "TP MỚI"] + unique_custs)
with col_h2: audit_cust = st.text_input("Tên khách cho Audit hiện tại", "VINEYARD VINES")

file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang quét toàn bộ thông số...", expanded=True) as status:
        target = extract_pdf_pro(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Đã quét xong: **{len(target['all_specs'])} Size** | Loại: **{target['category']}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                df_db['sim_score'] = cosine_similarity(target_vec, np.array([v for v in df_db['vector']])).flatten()
                
                # Logic Ưu tiên khách hàng
                tar_c = audit_cust.upper().strip()
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 3 if tar_c in str(x).upper() else (2 if sel_prio.upper() in str(x).upper() else 1))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(2)
                status.update(label="✅ Đã tìm thấy mẫu khớp nhất!", state="complete", expanded=False)

                # HIỂN THỊ KẾT QUẢ
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    st.info(f"📍 Mẫu {i+1}: {row['file_name']} (Khách: {row['customer_name']})")
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.image(row['image_url'], use_container_width=True)
                    with c2:
                        common_sizes = sorted(list(set(target['all_specs'].keys()) & set(row['spec_json'].keys())), key=lambda x: int(x) if x.isdigit() else x)
                        sel_s = st.selectbox(f"Chọn Size đối soát mẫu {i+1}:", common_sizes if common_sizes else list(target['all_specs'].keys()), key=f"s_{i}")
                        
                        if sel_s:
                            aud_d = target['all_specs'].get(sel_s, {})
                            lib_s = sel_s if sel_s in row['spec_json'] else list(row['spec_json'].keys())[0]
                            lib_d = row['spec_json'].get(lib_s, {})
                            
                            res_l = []
                            for k, v_aud in aud_d.items():
                                v_lib = lib_d.get(k, 0)
                                diff = round(v_aud - v_lib, 3)
                                res_l.append({"Vị trí (POM)": k, "Mẫu Gốc": v_lib, "Audit": v_aud, "Lệch": diff, "Kết quả": "✅" if abs(diff)<0.1 else "❌"})
                            st.table(pd.DataFrame(res_l))
                            
                            # Nút Excel
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine='xlsxwriter') as wr: pd.DataFrame(res_l).to_excel(wr, index=False)
                            st.download_button(f"📥 Tải Excel Mã {i+1}", buf.getvalue(), f"Audit_{row['file_name']}.xlsx", key=f"dl_{i}")
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại hàng", state="error")
        else: status.update(label="❌ Lỗi trích xuất thông số. Kiểm tra PDF.", state="error")
