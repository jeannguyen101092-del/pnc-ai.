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

st.set_page_config(layout="wide", page_title="AI Smart Auditor V103", page_icon="📏")

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

# ================= 3. UTILS & PARSER =================
def auto_detect_customer(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["REITMANS", "PENNINGTONS", "RW&CO"]): return "Reitmans"
    if "VINEYARD VINES" in t or "WHALE" in t: return "Vineyard Vines"
    return "Khác"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        # Xử lý hỗn số (ví dụ: 1 1/2)
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', txt)
        if mixed:
            g, n, d = mixed.groups()
            return float(g) + (float(n)/float(d))
        # Xử lý phân số (ví dụ: 1/2)
        frac = re.match(r'(\d+)/(\d+)', txt)
        if frac:
            n, d = frac.groups()
            return float(n)/float(d)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
        return float(nums[0]) if nums else 0
    except: return 0

def extract_pdf_v103(file):
    specs, img_bytes, category, cust_found = {}, None, "KHÁC", "Khác"
    try:
        file.seek(0)
        pdf_content = file.read()
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            if len(doc) > 0:
                img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                full_text = "".join([p.get_text() for p in doc])
                t_up = (full_text + file.name).upper()
                if any(x in t_up for x in ["PANT", "JEAN", "SKIRT", "QUẦN", "VÁY"]): category = "QUẦN/VÁY"
                elif any(x in t_up for x in ["SHIRT", "TOP", "TEE", "ÁO", "JACKET"]): category = "ÁO"
                cust_found = auto_detect_customer(full_text, file.name)
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    txt_tb = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "POM NAME", "SPEC"] if k in txt_tb) < 2: continue
                    
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if "POM NAME" in row_up: # Đặc thù Reitmans
                            n_col = row_up.index("POM NAME")
                            v_col = next((i for i, v in enumerate(row_up) if any(x in v for x in ["NEW", "SAMPLE", "SPEC"])), -1)
                        else:
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESC", "POSITION"]): n_col = i; break
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["NEW", "SAMPLE", "SPEC", "M", "32"]): v_col = i; break
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["TOL", "REF"]): continue
                                val = parse_val(df.iloc[d_idx, v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break
        return {"specs": specs, "img": img_bytes, "category": category, "customer": cust_found}
    except: return None

# ================= 4. MAIN =================
st.title("🔍 AI SMART AUDITOR - V103")

with st.sidebar:
    st.header("📂 KHO MASTER")
    try:
        res = supabase.table("ai_data").select("id", count="exact").execute()
        st.success(f"📊 Trong kho: **{res.count}** mẫu")
    except: st.info("Kho trống")
    
    new_files = st.file_uploader("Nạp Master", type="pdf", accept_multiple_files=True, key=f"up_{st.session_state.upload_id}")
    if new_files and st.button("🚀 NẠP KHO"):
        for f in new_files:
            data = extract_pdf_v103(f)
            if data and data['specs']:
                path = f"m_{re.sub(r'[^a-zA-Z0-9]', '_', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "customer": data['customer'], "prod_id": f.name.split('.')[0],
                    "vector": get_vector(data['img']), "spec_json": data['specs'], "image_url": url, "category": data['category']
                }).execute()
        st.session_state.upload_id += 1
        st.rerun()

c1, c2 = st.columns(2)
with c1: sel_cust = st.selectbox("🎯 Lọc Khách hàng:", ["Tất cả", "Reitmans", "Vineyard Vines"])
with c2: sel_prod = st.text_input("🆔 Mã Style:").strip().upper()

file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    target = extract_pdf_v103(file_audit)
    if target:
        st.info(f"✨ Phát hiện: **{target['customer']}** | Loại: **{target['category']}**")
        
        query = supabase.table("ai_data").select("*").eq("category", target['category'])
        if sel_cust != "Tất cả": query = query.eq("customer", sel_cust)
        db_res = query.execute()

        if db_res.data:
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            best_match, max_sim = None, -1.0
            
            for item in db_res.data:
                # ÉP KIỂU VECTOR ĐỂ TRÁNH LỖI TYPEERROR
                db_vector = np.array(item['vector']).astype(float)
                sim = float(cosine_similarity(t_vec, [db_vector]))
                if sim > max_sim:
                    max_sim = sim; best_match = item

            if best_match and max_sim > 0.5:
                st.subheader(f"📊 Kết quả (Độ giống: {max_sim:.1%})")
                col_a, col_b = st.columns(2)
                with col_a: st.image(target['img'], caption="BẢN AUDIT", use_container_width=True)
                with col_b: st.image(best_match['image_url'], caption=f"MASTER: {best_match['file_name']}", use_container_width=True)
                
                # Hiển thị bảng
                m_specs = best_match['spec_json']
                all_poms = sorted(set(list(target['specs'].keys()) + list(m_specs.keys())))
                rows = []
                for p in all_poms:
                    v_m, v_t = m_specs.get(p, 0), target['specs'].get(p, 0)
                    diff = round(v_t - v_m, 3)
                    res = "✅ Khớp" if abs(diff) < 0.125 else f"❌ Lệch ({diff})"
                    rows.append({"Vị trí đo": p, "Master": v_m, "Audit": v_t, "Kết quả": res})
                st.table(pd.DataFrame(rows))
            else:
                st.error("Không tìm thấy mẫu tương đồng.")
