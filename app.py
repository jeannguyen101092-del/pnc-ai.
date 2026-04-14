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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V103", page_icon="🏢")

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
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "QUẦN", "FLY-STITCH", "DENIM"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET"]
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
            p = s.split(); main_val = float(p[0])
            f_parts = p[1].split('/')
            return main_val + (float(f_parts[0]) / float(f_parts[1]))
        elif '/' in s:
            f_parts = s.split('/')
            return float(f_parts[0]) / float(f_parts[1])
        val = float(s)
        return val if val < 500 else 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE V103 (CỰC MẠNH) =================
def extract_pdf_v103(file):
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
                    if df.shape[1] < 2: continue
                    
                    n_col, size_cols = -1, {}
                    # Bước 1: Dò tìm hàng tiêu đề (quét 20 dòng đầu)
                    for r_idx, row in df.head(20).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # Tìm cột POM (Hỗ trợ hàng Denim: DIM, ID, DESCRIPTION)
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["SPEC", "DESC", "POM", "POS", "DIM", "ID", "REF"]): 
                                n_col = i; break
                        # Tìm các cột Size (Số hoặc chữ S/M/L)
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            cl = v.replace("SIZE", "").strip()
                            if cl.isdigit() or any(s == cl for s in ["S", "M", "L", "XL", "XS", "XXL", "28", "29", "30", "31", "32", "33", "34"]):
                                size_cols[i] = cl
                        if n_col != -1 and size_cols: break
                    
                    # Bước 2: Fallback (Nếu không thấy tiêu đề chuẩn, ép lấy cột đầu làm POM)
                    if n_col == -1:
                        n_col = 0
                        for i in range(1, df.shape[1]):
                            if sum(1 for val in df.iloc[:10, i] if parse_val_safe(val) > 0) >= 2:
                                size_cols[i] = f"Col_{i}"

                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val_safe(df.iloc[d_idx, s_col])
                                if len(pom) > 2 and val > 0 and not any(x in pom for x in ["DESCRIPTION", "SPEC", "POM", "DIM", "TOL"]):
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", "file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
        existing_files = [x['file_name'] for x in db_res.data]
    except: unique_custs, existing_files = [], []

    st.divider()
    new_cust = st.text_input("Nạp mẫu mới - Nhập tên khách hàng", placeholder="Ví dụ: Vineyard Vines")
    new_files = st.file_uploader("Nạp Techpack mới vào kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        success_count = 0
        for f in new_files:
            if f.name in existing_files:
                st.warning(f"⚠️ {f.name} đã tồn tại.")
                continue
            data = extract_pdf_v103(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": data['category'], "customer_name": new_cust.upper() or "UNKNOWN"}).execute()
                    success_count += 1
                except Exception as e: st.error(f"Lỗi DB: {e}")
            else: st.error(f"❌ File {f.name} không tìm thấy bảng thông số hợp lệ.")
        if success_count > 0: st.success(f"Đã nạp {success_count} file!"); st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR V103")
col_h1, col_h2 = st.columns(2)
with col_h1: sel_cust = st.selectbox("🎯 Chọn khách hàng ưu tiên hiển thị", ["Tất cả khách hàng", "TP MỚI"] + unique_custs)
with col_h2: tar_name = st.text_input("Nhập tên khách hàng cho Audit hiện tại", placeholder="Để AI tự tìm")

file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang quét thông số...", expanded=True) as status:
        target = extract_pdf_v103(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Nhận diện: **{target['category']}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                df_db['sim_score'] = cosine_similarity(target_vec, np.array([v for v in df_db['vector']])).flatten()
                
                # Logic ưu tiên: Khách đang nhập > TP MỚI > Dropdown > Khác
                f_tar = tar_name.upper() if tar_name.strip() else "NONE"
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 4 if f_tar in str(x).upper() else (3 if "TP MỚI" in str(x).upper() else (2 if sel_cust.upper() in str(x).upper() else 1)))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(5)
                
                status.update(label="✅ Đã tìm thấy mẫu tương đồng!", state="complete", expanded=False)
                
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
                                res_l = [{"POM": k, "Mẫu Gốc (S" + str(b_size) + ")": lib_d.get(k,0), "Audit (S" + str(sel_s) + ")": v, "Lệch": round(v-lib_d.get(k,0),3)} for k, v in aud_d.items()]
                                df_final = pd.DataFrame(res_l)
                                st.table(df_final)
                                
                                buf = io.BytesIO()
                                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                                    df_final.to_excel(writer, index=False)
                                st.download_button(label=f"📥 Tải Excel Mã {i+1}", data=buf.getvalue(), file_name=f"Audit_{row['file_name']}.xlsx", key=f"dl_{i}")
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại", state="error")
        else: status.update(label="❌ Không tìm thấy bảng thông số", state="error")
