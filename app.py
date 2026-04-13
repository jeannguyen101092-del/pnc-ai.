import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, json
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase = create_client(URL, KEY)
except:
    st.error("Kết nối Supabase thất bại!")

st.set_page_config(layout="wide", page_title="AI Smart Auditor V104", page_icon="📏")

if 'up_id' not in st.session_state: st.session_state.up_id = 0

# ================= 2. AI MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return vec.tolist()

# ================= 3. TRÍCH XUẤT DỮ LIỆU =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if '/' in txt: # Xử lý phân số (1 1/2)
            p = re.findall(r'\d+', txt)
            if len(p) == 2: return float(p[0])/float(p[1])
            if len(p) == 3: return float(p[0]) + (float(p[1])/float(p[2]))
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def extract_pdf_v104(file):
    specs, img_bytes, category, customer = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                text = "".join([p.get_text() for p in doc]).upper()
                if "REITMANS" in text: customer = "Reitmans"
                elif "VINEYARD VINES" in text: customer = "Vineyard Vines"
                category = "ÁO" if any(x in text + file.name.upper() for x in ["SHIRT", "TOP", "ÁO"]) else "QUẦN/VÁY"
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    txt_tb = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["CHEST", "WAIST", "POM NAME"] if k in txt_tb) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        # Logic riêng cho Reitmans
                        if "POM NAME" in row_up:
                            n_col = row_up.index("POM NAME")
                            v_col = next((i for i, v in enumerate(row_up) if any(x in v for x in ["NEW", "SAMPLE", "SPEC"])), -1)
                        else:
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i
                                if any(x in v for x in ["NEW", "SPEC", "M", "32"]): v_col = i
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or "TOL" in name: continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": customer}
    except: return None

# ================= 4. GIAO DIỆN CHÍNH =================
st.title("🔍 AI SMART AUDITOR - V104")

with st.sidebar:
    st.header("📂 KHO MASTER")
    try:
        res = supabase.table("ai_data").select("id", count="exact").execute()
        st.success(f"📊 Trong kho: **{res.count}** mẫu")
    except: st.info("Kho trống")
    
    new_files = st.file_uploader("Nạp Master PDF", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.up_id}")
       if new_files and st.button("🚀 NẠP VÀO KHO"):
        for f in new_files:
            data = extract_pdf_v104(f)
            if data and data['specs']:
                path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                # CHỈNH SỬA DÒNG NÀY ĐỂ TRÁNH LỖI API:
                try:
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "customer": data['customer'], 
                        "prod_id": str(f.name.split('.')[0]), # Đảm bảo là chuỗi
                        "vector": get_vector(data['img']), 
                        "spec_json": data['specs'], 
                        "image_url": url, 
                        "category": data['category']
                    }).execute()
                except Exception as e:
                    st.error(f"Lỗi Supabase: {e}")
        st.session_state.up_id += 1
        st.rerun()


# ĐỐI SOÁT
c1, c2 = st.columns(2)
with c1: sel_cust = st.selectbox("🎯 Khách hàng:", ["Tất cả", "Reitmans", "Vineyard Vines"])
with c2: sel_prod = st.text_input("🆔 Mã Style:").strip().upper()

file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    target = extract_pdf_v104(file_audit)
    if target:
        st.info(f"✨ Phát hiện: **{target['customer']}** | Loại: **{target['category']}**")
        
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        db_res = query.execute()

        if db_res.data:
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match, max_sim = None, -1.0
            
            for item in db_res.data:
                # --- SỬA LỖI ÉP KIỂU TẠI ĐÂY ---
                try:
                    # Nếu vector lưu dạng chuỗi, chuyển về list
                    raw_vec = item['vector']
                    if isinstance(raw_vec, str): raw_vec = json.loads(raw_vec)
                    db_vector = np.array(raw_vec).astype(float)
                    
                    sim = float(cosine_similarity(t_vec, [db_vector])[0][0])
                    if sim > max_sim:
                        max_sim = sim; best_match = item
                except: continue

            if best_match and max_sim > 0.5:
                st.subheader(f"📊 Kết quả (Độ tương đồng: {max_sim:.1%})")
                col_a, col_b = st.columns(2)
                with col_a: st.image(target['img'], use_container_width=True, caption="BẢN AUDIT")
                with col_b: st.image(best_match['image_url'], use_container_width=True, caption=f"MASTER: {best_match['file_name']}")
                
                # Bảng so sánh
                m_s = best_match['spec_json']
                t_s = target['specs']
                all_poms = sorted(set(list(t_s.keys()) + list(m_s.keys())))
                rows = [{"Vị trí": p, "Master": m_s.get(p,0), "Audit": t_s.get(p,0), 
                         "Lệch": round(t_s.get(p,0) - m_s.get(p,0), 3)} for p in all_poms]
                st.table(pd.DataFrame(rows))
