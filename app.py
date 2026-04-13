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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V97", page_icon="📏")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. AI UTILS & AUTO-DETECTION =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# Danh sách khách hàng định nghĩa sẵn để AI tự quét
CUSTOMERS_DB = ["REITMANS", "WALMART", "ADIDAS", "TARGET", "GAP", "LEVIS", "H&M"]

def detect_customer(text):
    """Tự động quét tên khách hàng từ văn bản PDF"""
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
    # Đếm từ khóa để nhận diện chính xác hơn
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

# ================= 3. TRÍCH XUẤT THÔNG MINH =================
def extract_pdf_v97(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        # AI tự quét Loại hàng và Khách hàng
        category = detect_category(full_text, file.name)
        customer = detect_customer(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if not any(x in str(tb).upper() for x in ["WAIST", "CHEST", "LENGTH"]): continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i; break
                    
                    if n_col != -1:
                        max_n = 0
                        for i in range(len(df.columns)):
                            if i == n_col: continue
                            cnt = sum(1 for val in df.iloc[:12, i] if parse_val(val) > 0)
                            if cnt > max_n: max_n = cnt; v_col = i
                            
                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if len(name) > 3 and val > 0: specs[name] = val
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None

# ================= 4. SIDEBAR (KHÔI PHỤC BỘ ĐẾM KHO) =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    
    # Hiển thị số lượng mẫu thực tế trong kho
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số mẫu trong kho", f"{res_count.count or 0} file")
    except: st.error("Lỗi kết nối database.")

    st.divider()
    st.info("💡 Hệ thống sẽ tự quét tên Khách hàng từ nội dung PDF.")
    
    new_files = st.file_uploader("Nạp file PDF vào kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            with st.spinner(f"Đang xử lý {f.name}..."):
                data = extract_pdf_v97(f)
                if data and data['specs']:
                    vec = get_vector(data['img'])
                    path = f"lib_{re.sub(r'[^A-Z]', '', f.name.upper())}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                        "image_url": url, "category": data['category'], 
                        "customer": data['customer'] # Lưu khách hàng do AI tự quét
                    }).execute()
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. ĐỐI SOÁT TỰ ĐỘNG =================
st.title("🔍 AI SMART AUDITOR V97")

file_audit = st.file_uploader("📤 Upload file cần kiểm tra đối soát", type="pdf")

if file_audit:
    with st.spinner("Đang trích xuất dữ liệu..."):
        target = extract_pdf_v97(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Phát hiện: **{target['category']}** | Khách hàng: **{target['customer']}**")
        
        # Tìm trong kho mẫu cùng chủng loại
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if res.data:
            target_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            matches = []
            for item in res.data:
                sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))
                matches.append({**item, "sim": sim})
            
            # Ưu tiên mẫu cùng khách hàng tự động quét được
            matches_sorted = sorted(matches, key=lambda x: (x['customer'] == target['customer'], x['sim']), reverse=True)
            top_3 = matches_sorted[:3]

            cols = st.columns(3)
            for i, m in enumerate(top_3):
                with cols[i]:
                    tag = "💎 CÙNG KHÁCH" if m['customer'] == target['customer'] else "🌐 KHÁC KHÁCH"
                    st.image(m['image_url'], caption=f"{tag}\n{m['file_name']} ({m['sim']:.1%})")

            # Tự động chọn mẫu khớp nhất để so sánh
            best_match = top_3[0]
            st.subheader(f"📊 ĐANG SO SÁNH VỚI: {best_match['file_name']}")
            
            audit_list = []
            for pom, val in target['specs'].items():
                m_val = best_match['spec_json'].get(pom, 0)
                diff = round(val - m_val, 3) if m_val else 0
                status = "✅ Khớp" if abs(diff) < 0.126 else f"❌ Lệch ({diff})"
                audit_list.append({"POM": pom, "File Mới": val, "Mẫu Gốc": m_val, "Kết quả": status})
            
            st.table(pd.DataFrame(audit_list))
            
            # Xuất Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(audit_list).to_excel(writer, index=False, sheet_name='Audit')
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Audit_{target['customer']}.xlsx")
