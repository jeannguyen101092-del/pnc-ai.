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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V109", page_icon="👖")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f8f9fa !important; color: #333 !important; font-weight: bold !important; }
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
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP"]
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "QUẦN", "FLY-STITCH", "DENIM", "JEAN"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO"]
    
    s_sk = sum(1 for k in skirt_keys if k in t)
    s_pa = sum(1 for k in pant_keys if k in t)
    s_sh = sum(1 for k in shirt_keys if k in t)
    
    m = max(s_sk, s_pa, s_sh)
    if m == 0: return "KHÁC"
    if m == s_sk: return "VÁY"
    if m == s_pa: return "QUẦN"
    return "ÁO"

def parse_val_denim(t):
    """Xử lý siêu chuẩn các số đo phân số Denim (VD: 14 1/2, 9 3/4)"""
    try:
        if not t: return 0
        txt = str(t).replace(',', '.').strip().lower()
        # Loại bỏ các đơn vị và chữ gây nhiễu
        if any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs", "date"]): return 0
        
        # Regex tìm số nguyên kèm phân số hoặc số thập phân
        match = re.search(r'(\d+)\s+(\d+)/(\d+)', txt) # Trường hợp "14 1/2"
        if match:
            return float(match.group(1)) + (float(match.group(2)) / float(match.group(3)))
            
        match = re.search(r'(\d+)/(\d+)', txt) # Trường hợp "1/2"
        if match:
            return float(match.group(1)) / float(match.group(2))
            
        match = re.search(r'(\d+\.\d+|\d+)', txt) # Trường hợp "32.5" hoặc "32"
        if match:
            val = float(match.group(1))
            return val if val < 250 else 0
        return 0
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE (DENIM MASTER LOGIC) =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Chụp ảnh trang đầu tiên làm mẫu nhận diện
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
                    
                    # QUY TRÌNH QUÉT TIÊU ĐỀ DENIM
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        # 1. Tìm cột Description (Tên thông số)
                        if "DESCRIPTION" in row_up:
                            n_col = row_up.index("DESCRIPTION")
                        
                        # 2. Tìm các cột Size (Chỉ lấy cột là Số lớn hoặc chữ Size)
                        for i, v in enumerate(row_up):
                            if i == n_col: continue
                            # BỎ QUA CÁC CỘT NHIỄU TRONG ẢNH
                            if any(x in v for x in ["TOL", "DIM", "TAGS", "LENGTH", "REF"]): continue
                            
                            clean_v = v.replace("SIZE", "").strip()
                            # Nếu tiêu đề là số (VD: 28, 30, 31, 32...)
                            if clean_v.isdigit() and int(clean_v) > 10:
                                size_cols[i] = clean_v
                            elif clean_v in ["S", "M", "L", "XL", "XS", "XXL"]:
                                size_cols[i] = clean_v
                        
                        if n_col != -1 and len(size_cols) > 0: break
                    
                    # 3. TRÍCH XUẤT DỮ LIỆU SỐ ĐO
                    if n_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom_name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                val = parse_val_denim(df.iloc[d_idx, s_col])
                                
                                # Lọc bỏ các dòng tiêu đề hoặc rác
                                if len(pom_name) > 4 and val > 0 and "DESCRIPTION" not in pom_name:
                                    all_specs[s_name][pom_name] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        db_res = supabase.table("ai_data").select("customer_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{db_res.count or 0} file")
        unique_custs = sorted(list(set([x['customer_name'] for x in db_res.data if x['customer_name']])))
    except: unique_custs = []

    st.divider()
    cust_name_save = st.text_input("Nhập tên khách hàng", value="Vineyard Vines")
    new_files = st.file_uploader("Nạp Techpack Denim mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                        "category": data['category'], "customer_name": cust_name_save.upper()
                    }).execute()
                except Exception as e: st.error(f"Lỗi DB: {e}")
        st.success("Nạp thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. ĐỐI SOÁT CHÍNH =================
st.title("👖 AI SMART AUDITOR - V109")
col_h1, col_h2 = st.columns(2)
with col_h1: sel_prio = st.selectbox("🎯 Chọn khách hàng ưu tiên", ["Tất cả khách hàng", "TP MỚI"] + unique_custs)
with col_h2: audit_cust_name = st.text_input("Tên khách cho bản Audit hiện tại", value="VINEYARD VINES")

file_audit = st.file_uploader("📤 Upload Techpack Denim Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang quét toàn bộ thông số Denim...", expanded=True) as status:
        target = extract_pdf_multi_size(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Đã quét xong: **{len(target['all_specs'])} Size** được tìm thấy.")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                df_db['sim_score'] = cosine_similarity(target_vec, np.array([v for v in df_db['vector']])).flatten()
                
                # Logic Ưu tiên khách hàng
                tar_c = audit_cust_name.upper().strip()
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 3 if tar_c in str(x).upper() else (2 if sel_prio.upper() in str(x).upper() else 1))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                status.update(label="✅ Đã tìm thấy mẫu tương đồng!", state="complete", expanded=False)

                # HIỂN THỊ KẾT QUẢ
                tabs = st.tabs([f"Mã {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    with tabs[i]:
                        c1, c2 = st.columns([1, 1.8])
                        with c1:
                            st.image(row['image_url'], use_container_width=True)
                            st.metric("Độ khớp AI", f"{row['sim_score']*100:.1f}%")
                            st.write(f"📌 Khách hàng: **{row['customer_name']}**")
                        with c2:
                            # TỰ ĐỘNG KHỚP SIZE
                            audit_sizes = sorted(list(target['all_specs'].keys()), key=lambda x: int(x) if x.isdigit() else x)
                            sel_s = st.selectbox(f"Chọn Size đối soát mẫu {i+1}:", audit_sizes, key=f"s_{i}")
                            
                            if sel_s:
                                aud_data = target['all_specs'].get(sel_s, {})
                                lib_json = row['spec_json']
                                # Lấy size tương ứng mẫu gốc (ưu tiên trùng tên, ko có lấy size đầu tiên)
                                b_size = sel_s if sel_s in lib_json else list(lib_json.keys())
                                lib_data = lib_json.get(b_size, {})
                                
                                st.write(f"📊 Đối soát: **Size {sel_s} (Audit)** vs **Size {b_size} (Gốc)**")
                                
                                results = []
                                for k, v_aud in aud_data.items():
                                    v_lib = lib_data.get(k, 0)
                                    diff = round(v_aud - v_lib, 4)
                                    status_txt = "✅ Khớp" if abs(diff) < 0.1 else f"❌ Lệch ({diff:+.2f})"
                                    results.append({"Vị trí đo (POM)": k, "Mẫu Gốc": v_lib, "Bản Audit": v_aud, "Kết quả": status_txt})
                                
                                st.table(pd.DataFrame(results))
                                
                                # Nút Excel
                                buf = io.BytesIO()
                                with pd.ExcelWriter(buf, engine='xlsxwriter') as wr: pd.DataFrame(results).to_excel(wr, index=False)
                                st.download_button(label=f"📥 Tải Excel Mã {i+1}", data=buf.getvalue(), file_name=f"Audit_{row['file_name']}.xlsx", key=f"dl_{i}")
            else: st.warning("❌ Không tìm thấy mẫu cùng loại hàng Denim trong kho.")
        else: st.error("❌ Không lấy được thông số. Hãy kiểm tra PDF có bảng không.")
