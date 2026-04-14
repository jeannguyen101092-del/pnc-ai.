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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V104", page_icon="🔍")

if 'up_key' not in st.session_state: st.session_state.up_key = 0

# ================= 2. AI UTILS (Nâng cấp nhận diện) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    # Mở rộng từ khóa để tránh nhận diện sai Váy thành Quần
    if any(x in t for x in ["DRESS", "SKIRT", "VÁY", "ĐẦM"]): return "VÁY"
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "TROUSER", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "JACKET", "TOP", "TEE", "ÁO"]): return "ÁO"
    return "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        val = float(v.split()[0]) + eval(v.split()[1]) if ' ' in v else (eval(v) if '/' in v else float(v))
        return val if 0.1 <= val < 150 else 0
    except: return 0

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
    return vec.astype(float).tolist()

# ================= 3. TRÍCH XUẤT PDF =================
def extract_pdf_v104(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text, file.name)
        # Tự động quét khách hàng (Express, Reitmans...)
        customer = "KHÁC"
        for c in ["EXPRESS", "REITMANS", "WALMART", "ADIDAS"]:
            if c in full_text.upper() or c in file.name.upper(): customer = c; break

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if not any(x in str(tb).upper() for x in ["WAIST", "CHEST", "LENGTH", "HIP"]): continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i; break
                    if n_col != -1:
                        for i in range(len(df.columns) - 1, n_col, -1):
                            cnt = sum(1 for val in df.iloc[:12, i] if parse_val(val) > 0)
                            if cnt >= 2: v_col = i; break
                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if len(name) > 3 and val > 0: specs[name] = val
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO")
    res_c = supabase.table("ai_data").select("customer", count="exact").execute()
    st.metric("Tổng số mẫu", f"{res_c.count or 0} file")
    unique_custs = sorted(list(set([item['customer'] for item in res_c.data if item['customer']])))
    
    st.divider()
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            data = extract_pdf_v104(f)
            if data and data['specs']:
                vec = get_vector(data['img'])
                path = f"lib_{re.sub(r'[^A-Z]', '', f.name.upper())}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category'], "customer": data['customer']}).execute()
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. MAIN FLOW =================
st.title("🔍 AI SMART AUDITOR - V104")

col1, col2 = st.columns(2)
with col1:
    filter_cust = st.selectbox("🎯 Lọc theo khách hàng:", ["TẤT CẢ (Tự động)"] + unique_custs)
with col2:
    file_audit = st.file_uploader("📤 Upload file đối soát", type="pdf")

if file_audit:
    target = extract_pdf_v104(file_audit)
    if target and target["specs"]:
        st.success(f"Phát hiện: **{target['category']}** | Khách hàng: **{target['customer']}**")
        
        # SỬA LỖI: Tìm kiếm linh hoạt hơn
        query = supabase.table("ai_data").select("*")
        if filter_cust != "TẤT CẢ (Tự động)":
            query = query.eq("customer", filter_cust)
        res = query.execute()
        
        if res.data:
            target_vec = np.array(get_vector(target['img']), dtype=np.float32).reshape(1, -1)
            matches = []
            for item in res.data:
                try:
                    db_vec = np.array(item['vector'], dtype=np.float32).reshape(1, -1)
                    sim = float(cosine_similarity(target_vec, db_vec))
                    # Ưu tiên mẫu cùng category và cùng khách hàng
                    score = sim + (0.5 if item['category'] == target['category'] else 0)
                    matches.append({**item, "sim": sim, "score": score})
                except: continue
            
            top_3 = sorted(matches, key=lambda x: x['score'], reverse=True)[:3]
            
            st.subheader("🖼️ MẪU TƯƠNG ĐỒNG NHẤT")
            cols = st.columns(len(top_3))
            for i, m in enumerate(top_3):
                with cols[i]:
                    st.image(m['image_url'], caption=f"{m['customer']} - {m['category']}\nGiống: {m['sim']:.1%}")

            best = top_3[0]
            st.subheader(f"📊 ĐỐI SOÁT VỚI: {best['file_name']}")
            audit_list = [{"POM": k, "Mới": v, "Gốc": best['spec_json'].get(k, 0), "KQ": "✅ Khớp" if abs(v - best['spec_json'].get(k, 0)) < 0.126 else f"❌ Lệch ({v - best['spec_json'].get(k, 0):+})"} for k, v in target['specs'].items()]
            st.table(pd.DataFrame(audit_list))
        else:
            st.warning("Không tìm thấy mẫu nào trong kho để so sánh.")
