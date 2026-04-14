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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V101", page_icon="🏢")

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
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "QUẦN", "FLY-STITCH", "CROTCH"]
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET"]
    s_sk, s_pa, s_sh = sum(1 for k in skirt_keys if k in t), sum(1 for k in pant_keys if k in t), sum(1 for k in shirt_keys if k in t)
    m = max(s_sk, s_pa, s_sh)
    if m == 0: return "KHÁC"
    if m == s_sk: return "VÁY"
    if m == s_pa: return "QUẦN"
    return "ÁO"

def parse_val(t):
    """Tính toán phân số an toàn không dùng eval()"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            main_num = float(parts[0])
            f_parts = parts[1].split('/')
            return main_num + (float(f_parts[0]) / float(f_parts[1]))
        elif '/' in v:
            f_parts = v.split('/')
            return float(f_parts[0]) / float(f_parts[1])
        return float(v)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT ĐA SIZE =================
def extract_pdf_v101(file):
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
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0 and not any(x in pom for x in ["SPECIFICATION", "DESCRIPTION", "POM"]):
                                    all_specs[s_name][pom] = val
                if all_specs: break
        return {"all_specs": all_specs, "img": img_bytes, "category": category}
    except Exception as e:
        st.error(f"Lỗi phân tích PDF: {e}")
        return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    count_res = supabase.table("ai_data").select("customer_name", count="exact").execute()
    st.metric("Tổng số mẫu hiện có", f"{count_res.count or 0} file")
    
    # Lấy danh sách khách hàng duy nhất để làm list sổ xuống
    unique_customers = sorted(list(set([x['customer_name'] for x in count_res.data]))) if count_res.data else []
    
    st.divider()
    cust_input_save = st.text_input("Nạp mẫu mới - Nhập tên khách hàng")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        progress_bar = st.progress(0)
        for i, f in enumerate(new_files):
            data = extract_pdf_v101(f)
            if data and data['all_specs']:
                try:
                    vec = get_image_vector(data['img'])
                    path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['all_specs'], 
                        "image_url": supabase.storage.from_(BUCKET).get_public_url(path), 
                        "category": data['category'], "customer_name": cust_input_save.upper() or "UNKNOWN"
                    }).execute()
                except Exception as e: st.error(f"Lỗi: {e}")
            progress_bar.progress((i + 1) / len(new_files))
        st.success("Nạp thành công!"); st.rerun()

# ================= 5. ĐỐI SOÁT - CHỌN KHÁCH & XUẤT EXCEL =================
st.title("🔍 AI SMART AUDITOR V101")

# LIST SỔ XUỐNG CHỌN TÊN KHÁCH HÀNG
selected_cust = st.selectbox("🎯 Chọn khách hàng để đối soát (Ưu tiên)", ["Tất cả khách hàng", "TP MỚI"] + unique_customers)

file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang đối soát dữ liệu...", expanded=True) as status:
        target = extract_pdf_v101(file_audit)
        if target and target["all_specs"]:
            st.write(f"✅ Nhận diện: **{target['category']}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                db_vecs = np.array([v for v in df_db['vector']])
                df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
                
                # Logic ưu tiên theo tên khách đã chọn
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 2 if selected_cust.upper() in str(x).upper() else 1)
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                
                status.update(label="✅ Đã tìm thấy mẫu tương đồng!", state="complete", expanded=False)
                
                tabs = st.tabs([f"Top {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
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
                                
                                # Tạo bảng dữ liệu đối soát
                                final_res = []
                                for pom, v_aud in aud_data.items():
                                    v_lib = lib_data.get(pom, 0)
                                    diff = round(v_aud - v_lib, 4)
                                    final_res.append({
                                        "POM": pom, "Mẫu Gốc (Lib)": v_lib, "Audit": v_aud, 
                                        "Lệch": diff, "Kết quả": "✅ Khớp" if abs(diff) < 0.05 else "❌ Lệch"
                                    })
                                
                                df_final = pd.DataFrame(final_res)
                                st.table(df_final)
                                
                                # --- NÚT XUẤT EXCEL ---
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    df_final.to_excel(writer, index=False, sheet_name='Audit_Report')
                                
                                st.download_button(
                                    label=f"📥 Tải Báo Cáo Excel (Mã {i+1})",
                                    data=buffer.getvalue(),
                                    file_name=f"Audit_Report_{row['file_name']}_{sel_s}.xlsx",
                                    mime="application/vnd.ms-excel",
                                    key=f"dl_{i}"
                                )
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại", state="error")
