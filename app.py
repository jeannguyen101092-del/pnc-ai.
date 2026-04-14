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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V103", page_icon="🔍")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. MODEL AI & CÔNG CỤ =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# Danh sách khách hàng để AI tự nhận diện
CUSTOMERS_DB = ["REITMANS", "EXPRESS", "VINEYARD VINES", "WALMART", "ADIDAS", "TARGET", "GAP", "LEVIS", "GUESS", "ZARA"]

def detect_customer(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    for cust in CUSTOMERS_DB:
        if cust in t: return cust
    return "KHÁC"

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {
        "VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM"],
        "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "QUẦN"],
        "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "COAT", "ÁO"]
    }
    scores = {k: sum(t.count(word) for word in v) for k, v in keywords.items()}
    detected = max(scores, key=scores.get)
    return detected if scores[detected] > 0 else "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 8 or not txt or any(x in txt for x in ["mm", "yd", "202"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        val = float(v.split()[0]) + eval(v.split()[1]) if ' ' in v else (eval(v) if '/' in v else float(v))
        return val if 0.1 <= val < 100 else 0
    except: return 0

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
    return vec.astype(float).tolist()

# ================= 3. TRÍCH XUẤT THÔNG MINH =================
def extract_pdf_v103(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text, file.name)
        customer = detect_customer(full_text, file.name)
        POM_KEYS = ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER", "THIGH", "RISE", "SLEEVE"]

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    flat_text = str(tb).upper()
                    if not any(x in flat_text for x in POM_KEYS): continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): n_col = i; break
                        if n_col != -1: break
                    
                    if n_col != -1:
                        # Dò từ phải qua trái để lấy cột thông số kích thước
                        for i in range(len(df.columns) - 1, n_col, -1):
                            cnt = sum(1 for val in df.iloc[:15, i] if 0.1 <= parse_val(val) <= 99)
                            if cnt >= 2: v_col = i; break
                            
                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if any(k in name for k in POM_KEYS) and 0.1 <= val < 100:
                                specs[name] = val
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO")
    try:
        res_c = supabase.table("ai_data").select("customer", count="exact").execute()
        st.metric("Tổng số mẫu trong kho", f"{res_c.count or 0} file")
        unique_custs = sorted(list(set([item['customer'] for item in res_c.data if item['customer']])))
    except: unique_custs = []

    st.divider()
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            data = extract_pdf_v103(f)
            if data and data['specs']:
                vec = get_vector(data['img'])
                path = f"lib_{re.sub(r'[^A-Z]', '', f.name.upper())}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                    "image_url": url, "category": data['category'], "customer": data['customer']
                }).execute()
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. MAIN FLOW (SỬA LỖI DÒNG 145) =================
st.title("🔍 AI SMART AUDITOR - V103")

col_f1, col_f2 = st.columns()
with col_f1:
    filter_cust = st.selectbox("🎯 Lọc theo khách hàng:", ["TẤT CẢ (Tự động)"] + unique_custs)
with col_f2:
    file_audit = st.file_uploader("📤 Upload file đối soát", type="pdf")

if file_audit:
    with st.spinner("Đang đối soát dữ liệu..."):
        target = extract_pdf_v103(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Phát hiện: **{target['category']}** | Khách hàng: **{target['customer']}**")
        
        # Truy vấn mẫu cùng chủng loại
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if filter_cust != "TẤT CẢ (Tự động)":
            query = query.eq("customer", filter_cust)
        res = query.execute()
        
        if res.data:
            target_vec = np.array(get_vector(target['img']), dtype=np.float32).reshape(1, -1)
            matches = []
            for item in res.data:
                try:
                    # SỬA LỖI TYPEERROR: Ép kiểu vector an toàn
                    db_vec = np.array(item['vector'], dtype=np.float32).reshape(1, -1)
                    sim = float(cosine_similarity(target_vec, db_vec))
                    matches.append({**item, "sim": sim})
                except: continue
            
            # Ưu tiên khách hàng hiện tại + độ giống nhau
            top_3 = sorted(matches, key=lambda x: (x['customer'] == target['customer'], x['sim']), reverse=True)[:3]
            
            # Hiển thị Top 3
            st.subheader("🖼️ MẪU TƯƠNG ĐỒNG TRONG KHO")
            cols = st.columns(len(top_3))
            for i, m in enumerate(top_3):
                with cols[i]:
                    tag = "💎 CÙNG KHÁCH" if m['customer'] == target['customer'] else "🌐 KHÁC KHÁCH"
                    st.image(m['image_url'], caption=f"{tag}\n{m['file_name']}\nGiống: {m['sim']:.1%}")

            # Tự động chọn mẫu tốt nhất để đối soát bảng
            best = top_3[0]
            st.subheader(f"📊 ĐỐI SOÁT VỚI: {best['file_name']}")
            
            audit_list = []
            for pom, val in target['specs'].items():
                m_val = best['spec_json'].get(pom, 0)
                diff = round(val - m_val, 3) if m_val else 0
                status = "✅ Khớp" if abs(diff) < 0.126 else f"❌ Lệch ({diff:+})"
                audit_list.append({"Vị trí đo (POM)": pom, "File Mới": val, "Mẫu Gốc": m_val, "Kết quả": status})
            
            st.table(pd.DataFrame(audit_list))
            
            # Nút Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(audit_list).to_excel(writer, index=False, sheet_name='Audit')
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Report_{target['customer']}.xlsx")
        else:
            st.warning("Chưa có dữ liệu mẫu để so sánh.")
    else:
        st.error("Không tìm thấy bảng thông số hợp lệ.")

st.caption("AI Smart Auditor V103 - Sửa lỗi đồng bộ & Tối ưu đối soát")
