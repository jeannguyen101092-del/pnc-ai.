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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V102", page_icon="🏢")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. CÔNG CỤ NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP"]
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "QUẦN", "FLY-STITCH"]
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
        s = match[0]
        if ' ' in s:
            p = s.split(); m = float(p[0]); f = p[1].split('/')
            return m + (float(f[0]) / float(f[1]))
        elif '/' in s:
            f = s.split('/')
            return float(f[0]) / float(f[1])
        return float(s)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE (NÂNG CẤP MẠNH) =================
def extract_pdf_v102(file):
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
                    df = pd.DataFrame(tb).fillna(""); n_col = -1; size_cols = {}
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # Tìm cột POM (Hỗ trợ thêm SPEC, DESC, POM, POSITION)
                        if any(x in v for x in ["SPEC", "DESC", "POM", "POS"]):
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["SPEC", "DESC", "POM", "POS"]): n_col = i; break
                        # Tìm Size
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            cl = v.replace("SIZE", "").strip()
                            if cl.isdigit() or cl in ["S", "M", "L", "XL", "XS", "XXL"]: size_cols[i] = cl
                        if n_col != -1 and size_cols: break
                    
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val_safe(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0: all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO (FIX LỖI) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_count = supabase.table("ai_data").select("customer_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_count.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_count.data if x['customer_name']])))
    except: unique_custs = []

    st.divider()
    cust_name = st.text_input("Nạp mẫu mới - Nhập tên khách hàng", placeholder="Ví dụ: Vineyard Vines")
    new_files = st.file_uploader("Nạp Techpack mới vào kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        count_success = 0
        existing = [x['file_name'] for x in supabase.table("ai_data").select("file_name").execute().data]
        
        for f in new_files:
            if f.name in existing:
                st.warning(f"Bỏ qua: {f.name} đã tồn tại.")
                continue
                
            with st.spinner(f"Đang xử lý {f.name}..."):
                data = extract_pdf_v102(f)
                if data and data['all_specs']:
                    try:
                        vec = get_image_vector(data['img'])
                        path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                        supabase.table("ai_data").insert({
                            "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                            "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                            "category": data['category'], "customer_name": cust_name.upper() or "UNKNOWN"
                        }).execute()
                        count_success += 1
                    except Exception as e: st.error(f"Lỗi Database: {e}")
                else:
                    st.error(f"⚠️ File {f.name} không có bảng thông số hợp lệ.")
        
        if count_success > 0:
            st.success(f"Đã nạp thành công {count_success} file!")
            st.session_state.up_key += 1
            st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR V102")
col_h1, col_h2 = st.columns(2)
with col_h1: sel_prio = st.selectbox("🎯 Chọn khách hàng ưu tiên hiển thị", ["Tất cả khách hàng"] + unique_custs)
with col_h2: tar_cust = st.text_input("Nhập tên khách hàng cho Audit hiện tại", placeholder="Để AI tự quét")

file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang xử lý đối soát...", expanded=True) as status:
        target = extract_pdf_v102(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Nhận diện: **{target['category']}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                df_db['sim_score'] = cosine_similarity(target_vec, np.array([v for v in df_db['vector']])).flatten()
                
                # Ưu tiên
                final_tar = tar_cust.upper() if tar_cust.strip() else "NONE"
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 3 if final_tar in str(x).upper() else (2 if "TP MỚI" in str(x).upper() else 1))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                
                status.update(label="✅ Đã tìm thấy các mẫu tương đồng!", state="complete", expanded=False)
                
                tabs = st.tabs([f"Mã {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    with tabs[i]:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(row['image_url'], use_container_width=True)
                            st.metric("Khớp AI", f"{row['sim_score']*100:.1f}%")
                            st.write(f"📌 Khách: **{row['customer_name']}**")
                        with c2:
                            aud_sizes = sorted(list(target['all_specs'].keys()))
                            sel_s = st.selectbox(f"Size đối soát {i+1}:", aud_sizes, key=f"s_{i}")
                            if sel_s:
                                aud_d = target['all_specs'].get(sel_s, {})
                                lib_j = row['spec_json']
                                b_size = sel_s if sel_s in lib_j else list(lib_j.keys())[0]
                                lib_d = lib_j.get(b_size, {})
                                res_l = [{"POM": k, "Gốc": lib_d.get(k,0), "Audit": v, "Lệch": round(v-lib_d.get(k,0),3)} for k, v in aud_d.items()]
                                st.table(pd.DataFrame(res_l))
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại hàng", state="error")
        else: status.update(label="❌ Không tìm thấy bảng thông số trong PDF", state="error")
