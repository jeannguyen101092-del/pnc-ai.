import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH KẾT NỐI =================
# Thay URL và KEY của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96 Pro", page_icon="👖")

# ================= 2. HÀM AI & PHÂN LOẠI THÔNG MINH =================
@st.cache_resource
def load_model():
    # Sử dụng mô hình ResNet18 để lấy đặc trưng hình ảnh
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def classify_garment(specs_dict, file_name=""):
    # Gộp tất cả thông số và tên file để quét từ khóa
    text_content = (" ".join(specs_dict.keys()) + " " + file_name).upper()
    
    # 1. QUẦN: Quét cực kỹ các từ khóa đặc thù
    pant_keywords = [
        "INSEAM", "OUTSEAM", "RISE", "LEG OPENING", "THIGH", 
        "CROTCH", "KNEE", "WAISTBAND", "PANT", "TROUSER", "SHORT"
    ]
    if any(k in text_content for k in pant_keywords):
        return "👖 QUẦN / CHÂN VÁY"
    
    # 2. ÁO: 
    top_keywords = ["BUST", "CHEST", "SHOULDER", "SLEEVE", "ARMHOLE", "NECK", "JACKET", "SHIRT"]
    if any(k in text_content for k in top_keywords):
        return "👕 ÁO / JACKET"
        
    return "👗 ĐẦM / KHÁC"

def parse_val(t):
    """Xử lý số và phân số (1/2, 3/4...) thường gặp trong ngành may"""
    try:
        if t is None or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v_str)) if '/' in v_str else float(v_str)
    except: return 0

def get_image_vector(img_bytes):
    """Chuyển ảnh thành vector để so sánh độ tương đồng"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. HÀM QUÉT PDF NHANH (TRANG 1 & TRANG POM) =================
def extract_pdf_smart_scan(file):
    all_specs, img_bytes = {}, None
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        # Luôn lấy ảnh trang đầu làm đại diện
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        
        # Tìm các trang có khả năng chứa bảng thông số
        target_pages = [0] # Mặc định kiểm tra cả trang 1
        keywords = ["POM", "MEASUREMENT", "SPEC", "DIMENSION", "WAIST", "INSEAM", "SIZE CHART"]
        for i in range(len(doc)):
            text = doc[i].get_text().upper()
            if any(k in text for k in keywords):
                if i not in target_pages: target_pages.append(i)
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for p_idx in target_pages:
                page = pdf.pages[p_idx]
                # table_settings giúp quét bảng không khung của Reitmans/AE
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text", 
                    "horizontal_strategy": "text", 
                    "snap_tolerance": 5
                })
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 3: continue
                    
                    desc_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        # Tìm cột Tên thông số
                        for i, v in enumerate(row):
                            if any(x in v for x in ["POM", "DESCRIPTION", "POSITION", "NAME"]):
                                desc_col = i; break
                        # Tìm các cột Size
                        for i, v in enumerate(row):
                            if i == desc_col or not v: continue
                            if v.isdigit() or v in ["XS","S","M","L","XL","XXL","3XL"]:
                                if not any(x in v for x in ["TOL", "+/-", "CODE"]): 
                                    size_cols[i] = v
                        if desc_col != -1 and size_cols: break
                    
                    if desc_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                # Loại bỏ tiêu đề lớn và dòng rác
                                if len(pom) > 3 and not (pom.isupper() and len(pom) > 25):
                                    val = parse_val(df.iloc[d_idx, s_col])
                                    if val > 0: all_specs[s_name][pom.upper()] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. GIAO DIỆN CHÍNH & LUỒNG XỬ LÝ =================
if 'uploader_key' not in st.session_state: 
    st.session_state['uploader_key'] = 0

with st.sidebar:
    st.header("🏢 KHO MẪU")
    # Lấy thông tin số lượng mẫu
    res_db = supabase.table("ai_data").select("id", "file_name", count="exact").execute()
    st.metric("Tổng tồn kho", f"{res_db.count if res_db.count else 0} mẫu")
    existing_files = [x['file_name'] for x in res_db.data] if res_db.data else []
    
    # File uploader tự xóa sau khi nạp nhờ key động
    files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=str(st.session_state['uploader_key']))
    if files and st.button("NẠP KHO"):
        for f in files:
            if f.name in existing_files:
                st.warning(f"Bỏ qua: {f.name} đã tồn tại."); continue
            data = extract_pdf_smart_scan(f)
            if data and data['all_specs']:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.session_state['uploader_key'] += 1
        st.rerun()

st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    target = extract_pdf_smart_scan(file_audit)
    if target and target["all_specs"]:
        # Nhận diện loại hàng từ size đầu tiên tìm thấy
        first_size_name = list(target['all_specs'].keys())[0]
        cat = classify_garment(target['all_specs'][first_size_name])
        st.info(f"📍 AI Nhận diện: **{cat}**")

        # So sánh tìm TOP 3
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            
            st.subheader("🎯 CHỌN MẪU ĐỐI SOÁT")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], use_container_width=True)
                    if st.button(f"Chọn: {row['file_name'][:15]}... ({row['sim']:.1%})", key=f"sel_{i}"):
                        st.session_state['active_idx'] = idx
            
            # Lấy mẫu thư viện được chọn
            best = top_3.loc[st.session_state.get('active_idx', top_3.index[0])]
            st.divider()
            st.subheader(f"📊 ĐỐI SOÁT CHI TIẾT: {best['file_name']}")
            
            # Chọn Size đối soát
            sel_s = st.selectbox("Chọn Size:", list(target['all_specs'].keys()))
            d_audit = target['all_specs'][sel_s]
            
            # Tìm size khớp trong kho
            lib_specs_all = best['spec_json']
            s_lib = sel_s if sel_s in lib_specs_all else list(lib_specs_all.keys())[0]
            d_lib = lib_specs_all[s_lib]
            
            report = []
            for pom, val in d_audit.items():
                ref = d_lib.get(pom, 0)
                diff = round(val - ref, 4)
                report.append({
                    "Thông số": pom, 
                    "Thực tế": val, 
                    "Mẫu kho": ref if ref != 0 else "N/A", 
                    "Lệch": diff if ref != 0 else 0,
                    "Kết quả": "✅ OK" if (ref != 0 and abs(diff) <= 0.25) else "⚠️ Ko khớp"
                })
            
            df_rep = pd.DataFrame(report)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)
            
            # Nút xuất Excel
            output = io.BytesIO()
            df_rep.to_excel(output, index=False, engine='xlsxwriter')
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Audit_{best['file_name']}_{sel_s}.xlsx")
    else:
        st.error("⚠️ Không tìm thấy bảng thông số trong các trang PDF đã quét.")
