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

try:
    supabase = create_client(URL, KEY)
except:
    st.error("Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Smart Auditor V101", page_icon="📏")

if 'upload_id' not in st.session_state: st.session_state.upload_id = 0

# ================= 2. MODEL AI =================
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

# ================= 3. UTILS & SPECIAL LOGIC FOR REITMANS =================
def auto_detect_customer(text):
    t = str(text).upper()
    if "REITMANS" in t: return "Reitmans"
    if "VINEYARD VINES" in t or "WHALE" in t: return "Vineyard Vines"
    if "NIKE" in t: return "Nike"
    if "ADIDAS" in t: return "Adidas"
    return "Khác"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def extract_pdf_v101(file):
    specs, img_bytes, category, cust_found = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        
        # 1. Quét sơ bộ bằng fitz để nhận diện Khách hàng & Loại hàng
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                t_up = (full_text + file.name).upper()
                if any(x in t_up for x in ["PANT", "JEAN", "SKIRT", "QUẦN", "VÁY", "BOTTOM"]): category = "QUẦN/VÁY"
                elif any(x in t_up for x in ["SHIRT", "TOP", "TEE", "ÁO", "JACKET", "HOODIE"]): category = "ÁO"
                cust_found = auto_detect_customer(full_text)
        
        # 2. Trích xuất bảng POM với logic riêng cho Reitmans
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    # Kiểm tra bảng POM
                    txt_tb = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "LENGTH", "HIP", "BUST", "POM NAME"] if k in txt_tb) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # --- ĐOẠN CODE RIÊNG CHO REITMANS ---
                        if cust_found == "Reitmans":
                            if "POM NAME" in row_up:
                                n_col = row_up.index("POM NAME")
                                v_col = next((i for i, v in enumerate(row_up) if "NEW" in v or "SAMPLE" in v), -1)
                        # --- LOGIC CHO CÁC HÃNG KHÁC ---
                        else:
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESCRIPTION", "DESC", "POSITION"]): n_col = i; break
                            for i, v in enumerate(row_up):
                                if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32", "34", "M", "S"]):
                                    if i != n_col: v_col = i; break
                        
                        # Nếu tìm thấy cột, tiến hành lấy số liệu
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARK", "COMMENT"]): continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": cust_found}
    except: return None

# ================= 4. SIDEBAR - QUẢN LÝ KHO =================
with st.sidebar:
    st.header("📂 KHO MASTER")
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        total_items = count_res.count if count_res.count is not None else 0
        st.success(f"📊 Tổng cộng trong kho: **{total_items}** mẫu")
    except: st.warning("Chưa kết nối kho.")

    st.divider()
    new_files = st.file_uploader("Nạp Techpack Master", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.upload_id}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        p_bar = st.progress(0); p_text = st.empty()
        for i, f in enumerate(new_files):
            pct = int((i + 1) / len(new_files) * 100)
            p_bar.progress(pct); p_text.text(f"Đang nạp {pct}%: {f.name}")
            data = extract_pdf_v101(f)
            if data and data['specs']:
                path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type":"image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "customer": data['customer'], "prod_id": f.name.split('.')[0],
                    "vector": get_vector(data['img']), "spec_json": data['specs'], "image_url": url, "category": data['category']
                }).execute()
        st.success("✅ Hoàn thành 100%!")
        st.session_state.upload_id += 1
        st.rerun()

# ================= 5. MAIN - ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V101")

c1, c2 = st.columns(2)
with c1: 
    sel_cust = st.selectbox("🎯 Lọc Khách hàng:", ["Tất cả", "Reitmans", "Vineyard Vines", "Nike", "Adidas"])
with c2: 
    sel_prod = st.text_input("🆔 Tìm mã Style (Production):").strip().upper()

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    target = extract_pdf_v101(file_audit)
    if target:
        st.info(f"✨ Hệ thống nhận diện: Khách hàng **{target['customer']}** | Loại hàng **{target['category']}**")
        
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        if sel_prod: query = query.ilike("prod_id", f"%{sel_prod}%")
        db_res = query.execute()
        
        if not db_res.data:
            st.warning(f"⚠️ Không tìm thấy mẫu Master khớp.")
        else:
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match, max_sim = None, -1.0
            for item in db_res.data:
                sim = float(cosine_similarity(t_vec, [item['vector']]))
                if sim > max_sim: max_sim = sim; best_match = item

            if best_match:
                st.subheader(f"📊 Đối soát: {target['customer']} vs {best_match['customer']} (Độ giống: {max_sim:.1%})")
                col_a, col_b = st.columns(2)
                with col_a: st.image(target['img'], caption="BẢN AUDIT", use_container_width=True)
                with col_b: st.image(best_match['image_url'], caption=f"BẢN MASTER: {best_match['file_name']}", use_container_width=True)
                
                # BẢNG SO SÁNH
                st.divider()
                m_specs = best_match['spec_json']
                all_poms = sorted(set(list(target['specs'].keys()) + list(m_specs.keys())))
                rows = []
                for p in all_poms:
                    v_m, v_t = m_specs.get(p, 0), target['specs'].get(p, 0)
                    diff = round(v_t - v_m, 3)
                    res = "✅ Khớp" if abs(diff) < 0.125 else f"❌ Lệch ({diff})"
                    rows.append({"Vị trí đo": p, "Master": v_m, "Audit": v_t, "Kết quả": res})
                st.table(pd.DataFrame(rows))
