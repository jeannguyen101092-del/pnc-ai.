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

# CSS làm đẹp
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    .percentage { color: #007bff; font-weight: bold; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. CÔNG CỤ NHẬN DIỆN THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def smart_detect(text, filename=""):
    """Nâng cấp nhận diện QUẦN, ÁO và VÁY triệt để"""
    t = (str(text) + " " + str(filename)).upper()
    
    # Từ khóa VÁY (Dress/Skirt): Hem, Sweep, Waist to Hem
    skirt_keys = ["SKIRT", "DRESS", "VÁY", "ĐẦM", "HEM", "SWEEP", "WAIST TO HEM", "BACK LENGTH", "BOTTOM WIDTH"]
    # Từ khóa QUẦN: Fly, Crotch, Inseam, Calf
    pant_keys = ["INSEAM", "THIGH", "RISE", "LEG OPENING", "PANT", "TROUSER", "QUẦN", "FLY-STITCH", "CALF", "CROTCH"]
    # Từ khóa ÁO: Chest, Shoulder, Sleeve, Neck
    shirt_keys = ["CHEST", "BUST", "SLEEVE", "SHOULDER", "NECK", "SHIRT", "ÁO", "JACKET", "BODY LENGTH"]
    
    # Tính điểm để quyết định loại nào mạnh nhất
    score_skirt = sum(1 for k in skirt_keys if k in t)
    score_pant = sum(1 for k in pant_keys if k in t)
    score_shirt = sum(1 for k in shirt_keys if k in t)
    
    max_score = max(score_skirt, score_pant, score_shirt)
    
    if max_score == 0: return "KHÁC"
    if max_score == score_skirt: return "VÁY"
    if max_score == score_pant: return "QUẦN"
    return "ÁO"

def extract_customer_v99(text):
    patterns = [r"(?i)(CUSTOMER|CLIENT|BUYER|KHÁCH HÀNG)[:\s]+([^\n\r]+)", r"(?i)BRAND[:\s]+([^\n\r]+)"]
    for p in patterns:
        match = re.search(p, text)
        if match: return match.group(2).strip().upper()[:30]
    return "UNKNOWN"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
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
        category, customer = smart_detect(full_text, file.name), extract_customer_v99(full_text)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna(""); n_col = -1; size_cols = {}
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM", "POSITION", "SPECIFICATION"]): n_col = i
                            elif v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "XS", "XXL"]): size_cols[i] = v
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

# ================= 4. SIDEBAR - NẠP KHO CÓ PROGRESS BAR =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{count_res.count or 0} file")
    except: pass
    st.divider()
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    if new_files and st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True):
        existing_names = [x['file_name'] for x in supabase.table("ai_data").select("file_name").execute().data]
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, f in enumerate(new_files):
            if f.name in existing_names: continue
            status_text.text(f"Đang quét: {f.name}...")
            data = extract_pdf_v99(f)
            if data and data['all_specs']:
                vec = get_image_vector(data['img'])
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": data['category'], "customer_name": data['customer']}).execute()
            progress_bar.progress((i + 1) / len(new_files))
        st.success("Cập nhật thành công!"); st.session_state.up_key += 1; st.rerun()

# ================= 5. ĐỐI SOÁT CÓ LOADING STATUS =================
st.title("🔍 AI SMART AUDITOR V99")
file_audit = st.file_uploader("📤 Upload Techpack Audit", type="pdf")

if file_audit:
    with st.status("🛠️ Đang quét nội dung PDF và đối soát AI...", expanded=True) as status:
        st.write("1. Trích xuất bảng thông số đa size...")
        target = extract_pdf_v99(file_audit)
        if target and target["all_specs"]:
            st.write(f"2. Loại hàng nhận diện: **{target['category']}** | Khách: **{target['customer']}**")
            res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
            if res.data:
                df_db = pd.DataFrame(res.data)
                target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
                db_vecs = np.array([v for v in df_db['vector']])
                df_db['sim_score'] = cosine_similarity(target_vec, db_vecs).flatten()
                tar_c = str(target['customer']).upper()
                df_db['priority'] = df_db['customer_name'].apply(lambda x: 3 if tar_c != "UNKNOWN" and tar_c in str(x).upper() else (2 if "TP MỚI" in str(x).upper() else 1))
                df_top = df_db.sort_values(by=['priority', 'sim_score'], ascending=[False, False]).head(3)
                status.update(label="✅ Đã tìm thấy mẫu tương đồng!", state="complete", expanded=False)
                
                tabs = st.tabs([f"Top {i+1}: {row['file_name']}" for i, row in df_top.iterrows()])
                for i, (idx, row) in enumerate(df_top.iterrows()):
                    with tabs[i]:
                        c_info, c_table = st.columns([1, 2])
                        with c_info:
                            st.image(row['image_url'], use_container_width=True)
                            st.metric("Độ khớp AI", f"{row['sim_score']*100:.1f}%")
                        with c_table:
                            common_sizes = list(set(target['all_specs'].keys()) & set(row['spec_json'].keys()))
                            if common_sizes:
                                sel_s = st.selectbox(f"Chọn Size đối soát {i+1}:", sorted(common_sizes), key=f"s_{i}")
                                lib_s, aud_s = row['spec_json'][sel_s], target['all_specs'][sel_s]
                                df_res = pd.DataFrame([{"Vị trí (POM)": k, "Mẫu Gốc": lib_s.get(k,0), "Audit": v, "Lệch": round(v-lib_s.get(k,0),4), "Kết quả": "✅ Khớp" if abs(v-lib_s.get(k,0))<0.01 else "❌ Lệch"} for k, v in aud_s.items()])
                                st.table(df_res)
            else: status.update(label="❌ Không tìm thấy mẫu cùng loại trong kho", state="error")
