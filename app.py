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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V99", page_icon="🏢")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. MODEL AI & NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect(text, filename=""):
    """Nhận diện Quần/Áo dựa trên các vị trí đo đặc trưng (POM)"""
    t = (str(text) + " " + str(filename)).upper()
    
    # Từ khóa đặc trưng của Quần và Áo
    is_pant = any(x in t for x in ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "TROUSER", "QUẦN"])
    is_shirt = any(x in t for x in ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET"])
    
    if is_pant and not is_shirt: return "QUẦN"
    if is_shirt and not is_pant: return "ÁO"
    return "KHÁC"

def extract_customer_v99(text):
    """Quét tên khách hàng mạnh mẽ hơn"""
    # Tìm kiếm các cụm từ phổ biến trong Techpack
    patterns = [
        r"(?i)(CUSTOMER|CLIENT|BUYER|KHÁCH HÀNG)[:\s]+([^\n\r]+)",
        r"(?i)BRAND[:\s]+([^\n\r]+)"
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            name = match.group(2).strip().upper()
            return name[:30] # Giới hạn độ dài tên
    return "UNKNOWN"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT PDF V99 =================
def extract_pdf_v99(file):
    all_specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        # Nhận diện quan trọng để phân loại Quần/Áo
        category = smart_detect(full_text, file.name)
        customer = extract_customer_v99(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna(""); n_col = -1; size_cols = {}
                    for r_idx, row in df.head(8).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM", "POSITION"]): n_col = i
                            elif v.isdigit() or any(s == v for s in ["S", "M", "L", "XL"]): size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0: all_specs[s_name][pom] = val
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer, "category": category}
    except: return None

# ================= 4. SIDEBAR - NẠP KHO (CHỐNG TRÙNG & PHÂN LOẠI) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    count_res = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Tổng số mẫu hiện có", f"{count_res.count or 0} file")
    
    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO"):
        # Lấy danh sách tên file đã có để tránh trùng
        existing_data = supabase.table("ai_data").select("file_name").execute().data
        existing_names = [x['file_name'] for x in existing_data]
        
        for f in new_files:
            if f.name in existing_names:
                st.warning(f"Bỏ qua: {f.name} đã tồn tại.")
                continue
                
            data = extract_pdf_v99(f)
            if data and data['all_specs']:
                vec = get_image_vector(data['img'])
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                
                # Insert vào DB: Đảm bảo cột category và customer không bị NULL
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                    "category": data['category'], # Đã fix
                    "customer_name": data['customer'] # Đã fix
                }).execute()
        st.success("Đã cập nhật kho mẫu!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. ĐỐI SOÁT ƯU TIÊN KHÁCH HÀNG =================
st.title("🔍 AI SMART AUDITOR V99")

file_audit = st.file_uploader("📤 Upload Techpack cần Audit", type="pdf")

if file_audit:
    with st.spinner("AI đang phân loại và tìm kiếm mẫu ưu tiên..."):
        target = extract_pdf_v99(file_audit)
    
    if target and target["all_specs"]:
        st.info(f"✨ Hệ thống nhận diện: **{target['category']}** | Khách hàng: **{target['customer']}**")
        
        # 1. LỌC: Chỉ tìm mẫu CÙNG LOẠI (ví dụ: Audit QUẦN thì chỉ tìm QUẦN trong kho)
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if res.data:
            df_db = pd.DataFrame(res.data)
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            db_vecs = np.array([v for v in df_db['vector']])
            
            # Tính % tương đồng hình ảnh
            df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
            
            # 2. LOGIC ƯU TIÊN: Trùng khách hàng (3) > TP MỚI (2) > Khác (1)
            def calc_prio(db_cust):
                db_cust = str(db_cust).upper()
                tar_cust = str(target['customer']).upper()
                if tar_cust != "UNKNOWN" and tar_cust in db_cust: return 3
                if "TP MỚI" in db_cust: return 2
                return 1
            
            df_db['priority'] = df_db['customer_name'].apply(calc_prio)
            
            # Sắp xếp lấy Top 3
            df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
            
            # Hiển thị kết quả chi tiết cho Top 3
            tabs = st.tabs([f"Top {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
            
            for i, (idx, row) in enumerate(df_top.iterrows()):
                with tabs[i]:
                    col_info, col_table = st.columns([1, 2])
                    with col_info:
                        st.image(row['image_url'], use_container_width=True)
                        st.write(f"📌 Khách: **{row['customer_name']}**")
                        st.write(f"🤖 Khớp hình ảnh: **{row['sim_score']*100:.1f}%**")
                    
                    with col_table:
                        common_sizes = list(set(target['all_specs'].keys()) & set(row['spec_json'].keys()))
                        if common_sizes:
                            sel_size = st.selectbox(f"Chọn Size đối soát (Mã {i+1}):", common_sizes, key=f"s_{i}")
                            lib_s, aud_s = row['spec_json'][sel_size], target['all_specs'][sel_size]
                            
                            res_list = []
                            for pom, v_aud in aud_s.items():
                                v_lib = lib_s.get(pom, 0)
                                diff = v_aud - v_lib
                                res_list.append({"Vị trí đo (POM)": pom, "Mẫu Gốc": v_lib, "Audit": v_aud, "Chênh lệch": round(diff, 2), "Kết quả": "✅ Khớp" if abs(diff)<0.1 else "❌ Lệch"})
                            
                            df_res = pd.DataFrame(res_list)
                            st.table(df_res)
                            
                            # Nút xuất Excel cho từng mã Top
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df_res.to_excel(writer, index=False)
                            st.download_button(f"📥 Tải Excel Top {i+1}", output.getvalue(), f"Report_{row['file_name']}.xlsx", key=f"dl_{i}")
        else:
            st.warning("Không tìm thấy mẫu nào cùng loại (Quần/Áo) trong kho để so sánh.")

