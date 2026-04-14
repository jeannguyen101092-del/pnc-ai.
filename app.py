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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V99.3", page_icon="🏢")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. CÔNG CỤ NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP", "WAIST TO HEM"]
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "TROUSER", "QUẦN", "FLY-STITCH", "CROTCH"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET", "BODY LENGTH"]
    s_sk, s_pa, s_sh = sum(1 for k in skirt_keys if k in t), sum(1 for k in pant_keys if k in t), sum(1 for k in shirt_keys if k in t)
    m = max(s_sk, s_pa, s_sh)
    if m == 0: return "KHÁC"
    if m == s_sk: return "VÁY"
    if m == s_pa: return "QUẦN"
    return "ÁO"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE =================
def extract_pdf_v99(file):
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
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["SPECIFICATION", "DESCRIPTION", "POM", "POSITION"]): n_col = i
                            elif v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "XS"]): size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0 and pom not in ["SPECIFICATION", "DESCRIPTION"]:
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi đọc file PDF: {e}")
        return None

# ================= 4. SIDEBAR - NẠP KHO (FIXED) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{count_res.count or 0} file")
    except Exception as db_err:
        st.error(f"Lỗi kết nối DB: {db_err}")
    
    st.divider()
    cust_input_save = st.text_input("Nhập tên khách hàng", "Vineyard Vines")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        res_existing = supabase.table("ai_data").select("file_name").execute()
        existing_names = [x['file_name'] for x in res_existing.data] if res_existing.data else []
        
        progress_bar = st.progress(0)
        for i, f in enumerate(new_files):
            if f.name in existing_names:
                st.warning(f"File {f.name} đã tồn tại trong kho.")
                continue
                
            data = extract_pdf_v99(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    
                    # 1. Upload ảnh lên Storage
                    storage_res = supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    # 2. Insert vào Table
                    db_res = supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                        "image_url": img_url, "category": data['category'], 
                        "customer_name": cust_input_save.upper()
                    }).execute()
                    
                except Exception as upload_err:
                    st.error(f"Lỗi khi nạp file {f.name}: {upload_err}")
            else:
                st.error(f"Không thể trích xuất bảng thông số từ file {f.name}. Kiểm tra định dạng PDF.")
            
            progress_bar.progress((i + 1) / len(new_files))
        
        st.success("Xử lý hoàn tất!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR V99.3")
cust_audit_input = st.text_input("Nhập tên khách hàng cần đối soát", "TP MỚI")
file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang quét nội dung...", expanded=True) as status:
        target = extract_pdf_v99(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Nhận diện: **{target['category']}** | Khách: **{cust_audit_input.upper()}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                db_vecs = np.array([v for v in df_db['vector']])
                df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
                tar_c = cust_audit_input.upper()
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 2 if tar_c in str(x).upper() else 1)
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                status.update(label="✅ Đã tìm thấy mẫu tương đồng!", state="complete", expanded=False)
                
                tabs = st.tabs([f"Top {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    with tabs[i]:
                        c_info, c_table = st.columns([1, 2])
                        with c_info:
                            st.image(row['image_url'], use_container_width=True)
                            st.metric("Khớp AI", f"{row['sim_score']*100:.1f}%")
                            st.write(f"📌 Khách: **{row['customer_name']}**")
                        with c_table:
                            common_sizes = sorted(list(set(target['all_specs'].keys()) & set(row['spec_json'].keys())))
                            if common_sizes:
                                sel_s = st.selectbox(f"Size đối soát {i+1}:", common_sizes, key=f"s_{i}")
                                lib_s, aud_s = row['spec_json'][sel_s], target['all_specs'][sel_s]
                                res_list = [{"POM": k, "Mẫu Gốc": lib_s.get(k,0), "Audit": v, "Lệch": round(v-lib_s.get(k,0),4), "Kết quả": "✅ Khớp" if abs(v-lib_s.get(k,0))<0.01 else "❌ Lệch"} for k, v in aud_s.items()]
                                st.table(pd.DataFrame(res_list))
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại", state="error")
