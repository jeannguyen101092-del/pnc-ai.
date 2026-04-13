import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V85", page_icon="🛡️")

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text):
    t = str(text).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "HOODIE", "SWEATER"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN"]): return "VÁY/ĐẦM"
    return "KHÁC"

def ultra_clean(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper().strip())

# ================= 3. HÀM TRÍCH XUẤT NÂNG CẤP V85 =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_table_score(df):
    """Tính điểm ưu tiên cho bảng thông số (POM)"""
    all_text = " ".join(df.astype(str).values.flatten()).upper()
    # Mở rộng bộ từ khóa nhận diện cho nhiều brand khác nhau
    pom_keywords = ["WAIST", "HIP", "RISE", "THIGH", "KNEE", "LEG", "INSEAM", "LENGTH", "CHEST", "SHOULDER", "SPEC", "SIZE", "MEASUREMENT", "TOL", "POM"]
    # Từ khóa phụ liệu cần loại bỏ
    bom_keywords = ["FABRIC", "THREAD", "BUTTON", "ZIPPER", "LABEL", "CONSUMPTION", "YARD", "MATERIAL"]
    
    score = sum(2 for word in pom_keywords if word in all_text)
    penalty = sum(3 for word in bom_keywords if word in all_text)
    
    # Đếm số lượng ô chứa giá trị số đo hợp lệ
    numeric_count = 0
    for col in df.columns[1:]:
        numeric_count += df[col].apply(lambda x: 1 if parse_val(x) > 0 else 0).sum()
    
    return score - penalty + (numeric_count * 0.5)

def extract_pdf_v85(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text)
        
        best_table_df = None
        highest_score = -999

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    current_score = get_table_score(df)
                    if current_score > highest_score:
                        highest_score = current_score
                        best_table_df = df

            # Nếu tìm thấy bảng có điểm cao nhất, tiến hành lấy dữ liệu
            if best_table_df is not None and highest_score > 0:
                df = best_table_df
                n_col, v_col = -1, -1
                for r_idx, row in df.head(15).iterrows():
                    row_up = [str(c).upper().strip() for c in row if c]
                    # Tìm cột Tên hạng mục
                    for i, v in enumerate(row_up):
                        if any(x in v for x in ["DESCRIPTION", "DESC", "POM", "MEASUREMENT"]): n_col = i; break
                    # Tìm cột Giá trị (Ưu tiên Sample Size/M/32...)
                    for i, v in enumerate(row_up):
                        if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "M", "32", "34", "36"]):
                            if i != n_col: v_col = i; break
                    
                    if n_col != -1 and v_col != -1:
                        for d_idx in range(r_idx + 1, len(df)):
                            d_row = df.iloc[d_idx]
                            name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                            if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARK"]): continue
                            val = parse_val(d_row[v_col])
                            if val > 0: specs[name] = val
                        break
        
        if not specs: return None
        return {"specs": specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    res_db = supabase.table("ai_data").select("file_name", "category").execute()
    data_lib = res_db.data if res_db.data else []
    st.info(f"Kho hiện tại: {len(data_lib)} file")

    st.divider()
    st.subheader("🚀 NẠP TECHPACK MỚI")
    new_files = st.file_uploader("Kéo thả PDF vào đây", type="pdf", accept_multiple_files=True)
    btn_nap = st.button("🚀 XÁC NHẬN NẠP KHO", use_container_width=True, disabled=not new_files)
    
    if btn_nap:
        for f in new_files:
            if any(d['file_name'] == f.name for d in data_lib):
                st.warning(f"⏩ Đã có: {f.name}"); continue
            with st.spinner(f"Đang phân tích {f.name}..."):
                data = extract_pdf_v85(f)
                if not data:
                    st.error(f"❌ Lỗi {f.name}: Không tìm thấy bảng thông số (POM) phù hợp."); continue
                
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category']}).execute()
                st.toast(f"✅ Đã nạp thành công: {f.name}")
        st.rerun()

# ================= 5. MAIN (SO SÁNH) =================
st.title("🔍 AI SMART AUDITOR - V85")
file_audit = st.file_uploader("Tải file cần đối soát", type="pdf", label_visibility="collapsed")

if file_audit:
    with st.spinner("Đang tìm bảng thông số..."):
        target = extract_pdf_v85(file_audit)
    
    if target:
        st.success(f"✨ Phát hiện: **{target['category']}** | {len(target['specs'])} vị trí đo.")
        db_res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        same_cat_data = db_res.data if db_res.data else []

        if not same_cat_data:
            st.warning("⚠️ Trong kho chưa có mẫu cùng chủng loại.")
        else:
            mode = st.radio("Chế độ so sánh:", ["🤖 Tự động (AI)", "👆 Chọn thủ công"], horizontal=True)
            sel_sample = None
            if mode == "🤖 Tự động (AI)":
                img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
                v_test = model_ai(transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
                matches = []
                for item in same_cat_data:
                    v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                    score = float(cosine_similarity(v_test, v_ref)[0][0])
                    matches.append({"item": item, "score": score})
                sel_sample = sorted(matches, key=lambda x: x['score'], reverse=True)[0]['item']
            else:
                choice = st.selectbox("Chọn mẫu gốc:", [d['file_name'] for d in same_cat_data])
                sel_sample = next(d for d in same_cat_data if d['file_name'] == choice)

            # HIỂN THỊ SONG SONG
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📄 ĐANG KIỂM")
                st.image(target["img"], use_container_width=True)
                st.table(pd.DataFrame([{"Vị trí": k, "Số đo": v} for k,v in target["specs"].items()]))
            with c2:
                st.markdown(f"### ✨ MẪU GỐC: {sel_sample['file_name']}")
                st.image(sel_sample['image_url'], use_container_width=True)
                
                ref_specs = sel_sample['spec_json']
                clean_ref = {ultra_clean(k): v for k, v in ref_specs.items()}
                rows = []
                for k, v in target["specs"].items():
                    v_ref = clean_ref.get(ultra_clean(k), 0)
                    diff = round(v - v_ref, 3)
                    res = "Khớp" if abs(diff) < 0.125 else "Lệch"
                    rows.append({"Vị trí so sánh": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": res})
                
                df_res = pd.DataFrame(rows)
                st.table(df_res.style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))
            
            # Xuất Excel
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False)
            st.download_button("📥 TẢI BÁO CÁO EXCEL", out.getvalue(), f"Audit_{sel_sample['file_name']}.xlsx", type="primary")
    else:
        st.error("❌ Không tìm thấy bảng thông số (POM) hợp lệ. Hãy kiểm tra PDF của bạn có phải là dạng văn bản không.")
