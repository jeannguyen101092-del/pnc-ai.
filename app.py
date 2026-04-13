import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, json
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase = create_client(URL, KEY)
except:
    st.error("Kết nối Supabase thất bại! Vui lòng kiểm tra URL/KEY.")

st.set_page_config(layout="wide", page_title="AI Smart Auditor V105", page_icon="📏")

if 'up_id' not in st.session_state: st.session_state.up_id = 0

# ================= 2. AI MODEL (TRÍCH XUẤT ĐẶC TRƯNG) =================
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

# ================= 3. TRÍCH XUẤT DỮ LIỆU PDF =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if '/' in txt:
            p = re.findall(r'\d+', txt)
            if len(p) == 2: return float(p[0])/float(p[1])
            if len(p) == 3: return float(p[0]) + (float(p[1])/float(p[2]))
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def extract_pdf_v105(file):
    specs, img_bytes, category, customer = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                text = "".join([p.get_text() for p in doc]).upper()
                if "REITMANS" in text: customer = "Reitmans"
                elif "VINEYARD VINES" in text or "WHALE" in text: customer = "Vineyard Vines"
                category = "ÁO" if any(x in text + file.name.upper() for x in ["SHIRT", "TOP", "ÁO", "JACKET"]) else "QUẦN/VÁY"
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    txt_tb = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["CHEST", "WAIST", "POM NAME", "SPEC", "LENGTH"] if k in txt_tb) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        # Logic Reitmans vs Hãng khác
                        if "POM NAME" in row_up:
                            n_col = row_up.index("POM NAME")
                            v_col = next((i for i, v in enumerate(row_up) if any(x in v for x in ["NEW", "SAMPLE", "SPEC"])), -1)
                        else:
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESC", "POM", "POSITION"]): n_col = i
                                if any(x in v for x in ["NEW", "SPEC", "M", "32", "SAMPLE"]): v_col = i
                        
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
st.title("🔍 AI SMART AUDITOR - V105")

with st.sidebar:
    st.header("📂 KHO MASTER")
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.success(f"📊 Trong kho: **{res_count.count if res_count.count is not None else 0}** mẫu")
    except: st.info("Kho trống")
    
    new_files = st.file_uploader("Nạp Master PDF", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.up_id}")
    if new_files and st.button("🚀 NẠP VÀO KHO"):
        p_bar = st.progress(0)
        for i, f in enumerate(new_files):
            data = extract_pdf_v105(f)
            if data and data['specs']:
                # 1. Upload ảnh lên Storage
                path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                # 2. Lưu vào Database (Sửa lỗi APIError)
                try:
                    supabase.table("ai_data").insert({
                        "file_name": f.name, 
                        "customer": data['customer'], 
                        "prod_id": f.name.split('.')[0], # Lấy mã trước dấu chấm
                        "vector": get_vector(data['img']), 
                        "spec_json": data['specs'], 
                        "image_url": url, 
                        "category": data['category']
                    }).execute()
                except Exception as e:
                    st.error(f"Lỗi DB với file {f.name}: {e}")
            p_bar.progress((i + 1) / len(new_files))
        
        st.success("Đã nạp kho xong!")
        st.session_state.up_id += 1
        st.rerun()

# KHU VỰC ĐỐI SOÁT
c1, c2 = st.columns(2)
with c1: sel_cust = st.selectbox("🎯 Lọc Khách hàng:", ["Tất cả", "Reitmans", "Vineyard Vines", "Nike", "Adidas"])
with c2: sel_prod = st.text_input("🆔 Tìm mã Style (Nếu có):").strip().upper()

file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    target = extract_pdf_v105(file_audit)
    if target:
        st.info(f"✨ Phát hiện: Khách hàng **{target['customer']}** | Loại: **{target['category']}**")
        
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        if sel_prod: query = query.ilike("prod_id", f"%{sel_prod}%")
        db_res = query.execute()

        if db_res.data:
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match, max_sim = None, -1.0
            
            for item in db_res.data:
                # ÉP KIỂU VECTOR ĐỂ TRÁNH LỖI TYPEERROR
                try:
                    raw_vec = item['vector']
                    if isinstance(raw_vec, str): raw_vec = json.loads(raw_vec)
                    db_vector = np.array(raw_vec).astype(float)
                    
                    sim = float(cosine_similarity(t_vec, [db_vector])[0][0])
                    if sim > max_sim:
                        max_sim = sim; best_match = item
                except: continue

            if best_match and max_sim > 0.5:
                st.subheader(f"📊 Kết quả đối soát (Độ giống: {max_sim:.1%})")
                col_a, col_b = st.columns(2)
                with col_a: 
                    st.markdown("### BẢN AUDIT")
                    st.image(target['img'], use_container_width=True)
                with col_b: 
                    st.markdown(f"### MASTER: {best_match['file_name']}")
                    st.image(best_match['image_url'], use_container_width=True)
                
                # BẢNG SO SÁNH THÔNG SỐ
                st.divider()
                m_s = best_match['spec_json']
                t_s = target['specs']
                all_poms = sorted(set(list(t_s.keys()) + list(m_s.keys())))
                
                res_table = []
                for p in all_poms:
                    v_m, v_t = m_s.get(p, 0), t_s.get(p, 0)
                    diff = round(v_t - v_m, 3)
                    status = "✅ Khớp" if abs(diff) < 0.125 else f"❌ Lệch ({diff})"
                    res_table.append({"Điểm đo (POM)": p, "Gốc (Master)": v_m, "Mới (Audit)": v_t, "Kết quả": status})
                
                st.table(pd.DataFrame(res_table))
            else:
                st.error("Không tìm thấy mẫu Master có hình ảnh tương đồng trong kho.")
        else:
            st.warning("Không tìm thấy dữ liệu Master phù hợp với điều kiện lọc.")

st.divider()
st.caption("AI Smart Auditor V105 - Tự động đối soát Techpack ngành may.")
