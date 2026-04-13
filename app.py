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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V100", page_icon="🔍")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. MODEL AI & CÔNG CỤ =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# Danh sách khách hàng để AI tự nhận diện (Hãy thêm khách hàng của bạn vào đây)
CUSTOMERS_DB = ["REITMANS", "WALMART", "ADIDAS", "TARGET", "GAP", "LEVIS", "H&M", "GUESS", "ZARA"]

def detect_customer(text):
    t = str(text).upper()
    for cust in CUSTOMERS_DB:
        if cust in t: return cust
    return "KHÁC"

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {
        "VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM", "GOWN"],
        "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "QUẦN"],
        "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "COAT", "ÁO"]
    }
    scores = {k: sum(t.count(word) for word in v) for k, v in keywords.items()}
    detected = max(scores, key=scores.get)
    return detected if scores[detected] > 0 else "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 8 or not txt or any(x in txt for x in ["mm", "yd", "date", "202"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        val = float(v.split()[0]) + eval(v.split()[1]) if ' ' in v else (eval(v) if '/' in v else float(v))
        return val if 0.1 <= val <= 180 else 0
    except: return 0

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad():
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT PDF V100 =================
def extract_pdf_v100(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text, file.name)
        customer = detect_customer(full_text)
        POM_KEYWORDS = ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER", "THIGH", "RISE", "INSEAM", "SLEEVE", "BUST", "NECK", "OPENING"]

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    flat_text = str(tb).upper()
                    if sum(1 for k in POM_KEYWORDS if k in flat_text) < 2: continue
                    
                    n_col = -1
                    v_col = -1
                    
                    # 1. Tìm cột Tên (Description)
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): 
                                n_col = i; break
                        if n_col != -1: break
                    
                    # 2. Dò cột Thông số (Ưu tiên cột bên phải n_col và có dải số từ 0.1 - 99)
                    if n_col != -1:
                        # Duyệt các cột từ PHẢI qua TRÁI để tránh bốc nhầm cột ID bên trái
                        for i in range(len(df.columns) - 1, n_col, -1):
                            cnt = sum(1 for val in df.iloc[:15, i] if 0.1 <= parse_val(val) <= 99)
                            if cnt >= 2: # Nếu cột có ít nhất 2 số đo hợp lệ
                                v_col = i; break

                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            
                            # Lọc rác: Tên phải chứa từ khóa ngành may và số phải < 100
                            is_measurement = any(k in name for k in POM_KEYWORDS)
                            if is_measurement and 0.1 <= val < 100:
                                specs[name] = val
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None
# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{res_count.count or 0} file")
    except: st.error("Database connection error.")

    st.divider()
    new_files = st.file_uploader("Nạp file PDF vào kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            data = extract_pdf_v100(f)
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

# ================= 5. LUỒNG ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR - V100")

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra đối soát", type="pdf")

if file_audit:
    with st.spinner("Đang AI quét dữ liệu PDF..."):
        target = extract_pdf_v100(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Phát hiện: **{target['category']}** | Khách hàng: **{target['customer']}**")
        
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if res.data:
            target_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            matches = []
            for item in res.data:
                sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))
                # Chuyển sim từ mảng sang số float đơn thuần
                matches.append({**item, "sim": float(sim[0][0])})
            
            # Ưu tiên cùng khách hàng + tương đồng nhất
            sorted_matches = sorted(matches, key=lambda x: (x['customer'] == target['customer'], x['sim']), reverse=True)
            top_3 = sorted_matches[:3]

            # HIỂN THỊ TOP 3 ẢNH
            st.subheader("🖼️ MẪU TƯƠNG ĐỒNG TRONG KHO")
            cols = st.columns(len(top_3))
            for i, m in enumerate(top_3):
                with cols[i]:
                    label = "💎 CÙNG KHÁCH" if m['customer'] == target['customer'] else "🌐 KHÁC KHÁCH"
                    st.image(m['image_url'], caption=f"{label}\n{m['file_name']}\nGiống: {m['sim']:.1%}")

            # TỰ ĐỘNG CHỌN MẪU TỐT NHẤT (Vị trí 0) ĐỂ SO SÁNH
            best = top_3[0]
            st.subheader(f"📊 ĐỐI SOÁT CHI TIẾT VỚI: {best['file_name']}")
            
            audit_list = []
            for pom, val in target['specs'].items():
                m_val = best['spec_json'].get(pom, 0)
                diff = round(val - m_val, 3) if m_val else 0
                status = "✅ Khớp" if abs(diff) < 0.126 else f"❌ Lệch ({diff:+})"
                if m_val == 0: status = "❓ Không có mẫu gốc"
                audit_list.append({"Vị trí đo (POM)": pom, "File Mới": val, "Mẫu Gốc": m_val, "Kết quả": status})
            
            df_audit = pd.DataFrame(audit_list)
            st.table(df_audit)
            
            # Nút Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_audit.to_excel(writer, index=False, sheet_name='Audit')
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Report_{target['customer']}.xlsx")
        else:
            st.warning("Trong kho chưa có mẫu cùng chủng loại để đối soát.")
    else:
        st.error("⚠️ Không tìm thấy bảng thông số trong PDF này. Hãy kiểm tra lại file!")

st.divider()
st.caption("AI Smart Auditor V100 - Final Version")
