import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Smart Auditor V100", page_icon="📏")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. CÔNG CỤ AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

CUSTOMERS_DB = ["REITMANS", "WALMART", "ADIDAS", "TARGET", "GAP", "LEVIS", "H&M", "GUESS", "ZARA"]

def detect_customer(text):
    t = str(text).upper()
    for cust in CUSTOMERS_DB:
        if cust in t: return cust
    return "KHÁC"

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {
        "VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM"],
        "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "QUẦN"],
        "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "ÁO"]
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

# ================= 3. TRÍCH XUẤT THÔNG MINH (SỬA LỖI KHÔNG HIỂN THỊ) =================
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

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Nới lỏng điều kiện tìm bảng: Chỉ cần thấy từ khóa đo đạc cơ bản
                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if not any(x in flat_text for x in ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER"]): continue
                    
                    n_col, v_col = -1, -1
                    # Dò cột Tên POM
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION", "ITEM"]):
                                n_col = i; break
                        if n_col != -1: break
                    
                    # Dò cột Giá trị (Cột có nhiều số đo nhất)
                    if n_col != -1:
                        max_n = 0
                        for i in range(len(df.columns)):
                            if i == n_col: continue
                            cnt = sum(1 for val in df.iloc[:15, i] if parse_val(val) > 0)
                            if cnt > max_n: max_n = cnt; v_col = i
                            
                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if len(name) > 3 and val > 0 and not any(x in name for x in ["POM", "DESC"]):
                                specs[name] = val
                if specs: break # Đã lấy được bảng thì dừng lại
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO MẪU")
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{res_count.count or 0} file")
    except: st.error("Database chưa sẵn sàng.")

    st.divider()
    new_files = st.file_uploader("Nạp file PDF mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    
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

# ================= 5. MAIN FLOW (LUÔN HIỂN THỊ KẾT QUẢ) =================
st.title("🔍 AI SMART AUDITOR - V100")

file_audit = st.file_uploader("📤 Upload file PDF đối soát", type="pdf")

if file_audit:
    with st.spinner("Đang AI quét dữ liệu..."):
        target = extract_pdf_v100(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Nhận diện: **{target['category']}** | Khách hàng: **{target['customer']}**")
        
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if res.data:
            target_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            matches = []
            for item in res.data:
                sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))
                matches.append({**item, "sim": sim})
            
            # Ưu tiên cùng khách hàng + tương đồng nhất
            sorted_matches = sorted(matches, key=lambda x: (x['customer'] == target['customer'], x['sim']), reverse=True)
            top_3 = sorted_matches[:3]

            # --- HIỂN THỊ TOP 3 MẪU TƯƠNG ĐỒNG ---
st.subheader("🖼️ MẪU TƯƠNG ĐỒNG TRONG KHO")
if top_3:
    cols = st.columns(len(top_3))
    for i, m in enumerate(top_3):
        with cols[i]:
            # Đảm bảo m là dictionary trước khi truy cập
            img_url = m.get('image_url', '')
            cust_name = m.get('customer', 'KHÁC')
            sim_score = m.get('sim', 0)
            
            st.image(img_url, caption=f"{cust_name} - {sim_score:.1%}")

    # --- TỰ ĐỘNG LẤY MẪU TỐT NHẤT ĐỂ SO SÁNH ---
    best = top_3[0] # Lấy mẫu đầu tiên trong danh sách đã sắp xếp
    st.subheader(f"📊 ĐỐI SOÁT VỚI: {best['file_name']}")
    
    audit_data = []
    for pom, val in target['specs'].items():
        m_val = best['spec_json'].get(pom, 0)
        diff = round(val - m_val, 3) if m_val else 0
        status = "✅ Khớp" if abs(diff) < 0.126 else f"❌ Lệch ({diff:+})"
        audit_data.append({"POM": pom, "Mới": val, "Gốc": m_val, "Kết quả": status})
    
    st.table(pd.DataFrame(audit_data))
else:
    st.warning("Không tìm thấy mẫu tương đồng để đối soát.")
            
            # Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(audit_list).to_excel(writer, index=False, sheet_name='Audit')
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Audit_{target['customer']}.xlsx")
        else:
            st.warning("Trong kho chưa có mẫu cùng chủng loại.")
    elif target:
        st.error("⚠️ Không tìm thấy bảng thông số trong PDF. Vui lòng kiểm tra lại file!")
