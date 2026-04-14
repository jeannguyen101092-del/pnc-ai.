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

# CSS làm đẹp
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
    # Bộ từ khóa nhận diện loại hàng
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP", "WAIST TO HEM"]
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "TROUSER", "QUẦN", "FLY-STITCH", "CROTCH"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET", "BODY LENGTH"]
    
    s_sk = sum(1 for k in skirt_keys if k in t)
    s_pa = sum(1 for k in pant_keys if k in t)
    s_sh = sum(1 for k in shirt_keys if k in t)
    
    m = max(s_sk, s_pa, s_sh)
    if m == 0: return "KHÁC"
    if m == s_sk: return "VÁY"
    if m == s_pa: return "QUẦN"
    return "ÁO"

def extract_customer_name(text):
    patterns = [r"(?i)(CUSTOMER|CLIENT|BUYER|KHÁCH HÀNG)[:\s]+([^\n\r]+)"]
    for p in patterns:
        match = re.search(p, text)
        if match: return match.group(2).strip().upper()[:30]
    return "UNKNOWN"

def parse_val_safe(t):
    """Sửa lỗi: Tính toán số đo an toàn, fix lỗi 'free variable V'"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        
        s = match[0] # Lấy chuỗi số đo đầu tiên
        if ' ' in s: # Số hỗn hợp: 27 1/2
            parts = s.split()
            main_val = float(parts[0])
            frac_parts = parts[1].split('/')
            return main_val + (float(frac_parts[0]) / float(frac_parts[1]))
        elif '/' in s: # Phân số: 1/2
            frac_parts = s.split('/')
            return float(frac_parts[0]) / float(frac_parts[1])
        return float(s)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE =================
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
        customer_scanned = extract_customer_name(full_text)
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna(""); n_col = -1; size_cols = {}
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        if any(x in v for x in ["SPECIFICATION", "DESCRIPTION", "POM", "POSITION"]):
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["SPECIFICATION", "DESCRIPTION", "POM", "POSITION"]): n_col = i; break
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            clean_v = v.replace("SIZE", "").strip()
                            if clean_v.isdigit() or clean_v in ["S", "M", "L", "XL", "XS", "XXL"]: size_cols[i] = clean_v
                        if n_col != -1 and size_cols: break
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val_safe(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0 and pom not in ["SPECIFICATION", "DESCRIPTION", "POM"]:
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category, "customer": customer_scanned}
    except: return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_customers = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
    except:
        st.error("Lỗi Database.")
        unique_customers = []

    st.divider()
    cust_input_save = st.text_input("Nạp mẫu mới - Nhập tên khách hàng", placeholder="Ví dụ: VINEYARD VINES")
    new_files = st.file_uploader("Nạp Techpack mới vào kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        existing_names = [x['file_name'] for x in supabase.table("ai_data").select("file_name").execute().data]
        progress_bar = st.progress(0)
        for i, f in enumerate(new_files):
            if f.name in existing_names:
                st.warning(f"⚠️ Bỏ qua: {f.name} đã có trong kho.")
                continue
            data = extract_pdf_v102(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    final_cust = cust_input_save.upper() if cust_input_save.strip() else data['customer']
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                        "category": data['category'], "customer_name": final_cust
                    }).execute()
                except Exception as e: st.error(f"Lỗi nạp {f.name}: {e}")
            progress_bar.progress((i + 1) / len(new_files))
        st.success("Nạp thành công!"); st.rerun()

# ================= 5. ĐỐI SOÁT - TOP 5 & ƯU TIÊN =================
st.title("🔍 AI SMART AUDITOR V102")

col_head1, col_head2 = st.columns(2)
with col_head1:
    selected_prio = st.selectbox("🎯 Chọn khách hàng ưu tiên hiển thị", ["Tất cả khách hàng", "TP MỚI"] + unique_customers)
with col_head2:
    cust_audit_name = st.text_input("Nhập tên khách hàng cho file Audit hiện tại", placeholder="Để trống nếu muốn AI tự quét")

file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang xử lý đối soát...", expanded=True) as status:
        target = extract_pdf_v102(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Nhận diện: **{target['category']}** | Khách (PDF): **{target['customer']}**")
            
            # 1. Lọc đúng loại hàng (Quần chỉ so Quần, Áo chỉ so Áo)
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                db_vecs = np.array([v for v in df_db['vector']])
                df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
                
                # 2. Logic ưu tiên khách hàng
                tar_name_final = cust_audit_name.upper() if cust_audit_name.strip() else target['customer']
                def calc_priority(db_cust):
                    db_cust = str(db_cust).upper()
                    if tar_name_final != "UNKNOWN" and tar_name_final in db_cust: return 4 # Ưu tiên 1: Cùng khách vừa nhập/quét
                    if selected_prio != "Tất cả khách hàng" and selected_prio.upper() in db_cust: return 3 # Ưu tiên 2: Khách chọn ở Dropdown
                    if "TP MỚI" in db_cust: return 2 # Ưu tiên 3: Khách trọng điểm
                    return 1
                
                df_db['priority'] = df_db['customer_name'].apply(calc_priority)
                # Sắp xếp và lấy TOP 5
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(5)
                
                status.update(label="✅ Đã tìm thấy các mẫu tương đồng!", state="complete", expanded=False)
                
                # 3. Hiển thị kết quả dạng Tabs cho 5 mã
                tabs = st.tabs([f"Mã {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    with tabs[i]:
                        c_info, c_table = st.columns(2)
                        with c_info:
                            st.image(row['image_url'], use_container_width=True)
                            st.metric("Độ khớp AI", f"{row['sim_score']*100:.1f}%")
                            st.write(f"📌 Khách hàng: **{row['customer_name']}**")
                        with c_table:
                            audit_sizes = sorted(list(target['all_specs'].keys()))
                            sel_s = st.selectbox(f"Chọn Size đối soát (Mã {i+1}):", audit_sizes, key=f"s_{i}")
                            if sel_s:
                                aud_data = target['all_specs'].get(sel_s, {})
                                lib_json = row['spec_json']
                                best_lib_size = sel_s if sel_s in lib_json else list(lib_json.keys())[0]
                                lib_data = lib_json.get(best_lib_size, {})
                                
                                st.write(f"📊 Đối soát: **Audit (S{sel_s})** vs **Gốc (S{best_lib_size})**")
                                final_res = [{"POM": k, "Gốc": lib_data.get(k,0), "Audit": v, "Lệch": round(v-lib_data.get(k,0),4), "Kết quả": "✅ Khớp" if abs(v-lib_data.get(k,0))<0.05 else "❌ Lệch"} for k, v in aud_data.items()]
                                df_final = pd.DataFrame(final_res)
                                st.table(df_final)
                                
                                # --- NÚT XUẤT EXCEL ---
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    df_final.to_excel(writer, index=False, sheet_name='Report')
                                st.download_button(label=f"📥 Tải Excel Mã {i+1}", data=buffer.getvalue(), file_name=f"Audit_{row['file_name']}.xlsx", mime="application/vnd.ms-excel", key=f"dl_{i}")
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại hàng", state="error")
