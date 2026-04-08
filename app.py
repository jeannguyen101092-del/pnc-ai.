import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. CẤU HÌNH & KẾT NỐI =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Kết nối Database thất bại: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V33", page_icon="📊")

# ================= 2. HỆ THỐNG AI VISION =================
@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad(): 
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

def to_excel(df_details, df_pom):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_details.to_excel(writer, sheet_name='Chi_Tiet', index=False)
        df_pom.to_excel(writer, sheet_name='POM', index=False)
    return output.getvalue()

# ================= 3. TRÍCH XUẤT TECHPACK =================
def deep_detail_inspection(text):
    t = str(text).upper()
    det = {}
    if any(x in t for x in ['JACKET', 'VEST', 'COAT']): det['Loại'] = "🧥 Áo Khoác"
    elif 'DRESS' in t: det['Loại'] = "👗 Đầm"
    elif 'PANT' in t or 'TROUSER' in t: det['Loại'] = "👖 Quần"
    else: det['Loại'] = "📦 Khác"
    det['Phụ liệu'] = "Dây kéo (Zipper)" if 'ZIPPER' in t else "Nút/Chun"
    return det

def extract_techpack(pdf_file):
    specs, img, raw_text = {}, None, ""
    pom_keys = ['WAIST', 'HIP', 'CHEST', 'BUST', 'LENGTH', 'SHOULDER', 'SLEEVE']
    try:
        pdf_bytes = pdf_file.read()
        # Trích xuất ảnh (trang 1)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        # Trích xuất bảng
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "")
                for tb in page.extract_tables():
                    for row in tb:
                        cells = [str(c).strip() for c in row if c]
                        if len(cells) < 2: continue
                        k_upper = str(cells[0]).upper()
                        if any(k in k_upper for k in pom_keys):
                            val = re.findall(r"\d+\.?\d*", " ".join(cells[1:]))
                            if val: specs[k_upper] = val[0]
        return {"spec": specs, "img": img, "details": deep_detail_inspection(raw_text)}
    except: return None

# ================= 4. GIAO DIỆN CHÍNH =================
st.title("📊 AI FASHION AUDITOR V33")

res = supabase.table("ai_data").select("*").execute()
samples = res.data if res else []

with st.sidebar:
    st.header("⚙️ Quản lý kho")
    st.metric("MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF gốc", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 NẠP DỮ LIỆU & ẢNH"):
        for f in up_pdfs:
            d = extract_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                # 1. Upload ảnh lên Storage
                img_path = f"{ma}.png"
                try:
                    supabase.storage.from_(BUCKET).upload(img_path, d['img'], {"content-type":"image/png", "upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
                except: img_url = None
                
                # 2. Lưu Database
                vec = get_vector(d['img'])
                supabase.table("ai_data").upsert({
                    "file_name": ma, "vector": vec, "spec_json": d['spec'], 
                    "details": d['details'], "image_url": img_url
                }).execute()
        st.success("Đã nạp thành công!")
        st.rerun()

# --- ĐỐI CHIẾU ---
sample_list = ["--- TỰ ĐỘNG TÌM KIẾM (AI) ---"] + [s['file_name'] for s in samples]
selected_code = st.selectbox("🎯 1. Chọn mã đối chứng:", sample_list)
test_files = st.file_uploader("📂 2. Tải Techpack kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_techpack(t_file)
        if not data_test: continue

        with st.expander(f"📑 KẾT QUẢ: {t_file.name}", expanded=True):
            best_match, sim_score = None, 0.0
            v_test = get_vector(data_test['img'])
            
            if v_test:
                vt_np = np.array(v_test).reshape(1, -1)
                if selected_code == "--- TỰ ĐỘNG TÌM KIẾM (AI) ---":
                    scores = []
                    valid_cands = []
                    for s in samples:
                        v_raw = s.get('vector')
                        if isinstance(v_raw, str): v_raw = json.loads(v_raw)
                        if v_raw:
                            score = float(cosine_similarity(vt_np, np.array(v_raw).reshape(1, -1)))
                            scores.append(score); valid_cands.append(s)
                    if scores:
                        idx = np.argmax(scores)
                        best_match, sim_score = valid_cands[idx], scores[idx]
                else:
                    best_match = next((s for s in samples if s['file_name'] == selected_code), None)
                    v_raw = best_match.get('vector') if best_match else None
                    if isinstance(v_raw, str): v_raw = json.loads(v_raw)
                    if v_raw: sim_score = float(cosine_similarity(vt_np, np.array(v_raw).reshape(1, -1)))

            if best_match:
                st.info(f"✅ Khớp mã: **{best_match['file_name']}** | Độ tương đồng: **{sim_score:.1%}**")
                
                # Hiển thị bảng
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🔍 Chi tiết")
                    g_det = best_match.get('details', {})
                    if isinstance(g_det, str): g_det = json.loads(g_det)
                    df_det = pd.DataFrame([{"Hạng mục": k, "Test": v, "Gốc": g_det.get(k, "-")} for k, v in data_test['details'].items()])
                    st.table(df_det)
                with c2:
                    st.subheader("📏 Thông số POM")
                    g_spec = best_match.get('spec_json', {})
                    if isinstance(g_spec, str): g_spec = json.loads(g_spec)
                    all_k = sorted(set(list(data_test['spec'].keys()) + list(g_spec.keys())))
                    df_pom = pd.DataFrame([{"Thông số": k, "Test": data_test['spec'].get(k,"-"), "Gốc": g_spec.get(k,"-")} for k in all_k])
                    st.table(df_pom)
                
                # Hiển thị hình ảnh
                st.divider()
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.image(data_test['img'], caption="Ảnh File Test", use_container_width=True)
                with col_i2:
                    if best_match.get('image_url'):
                        st.image(best_match['image_url'], caption="Ảnh File Gốc", use_container_width=True)
                    else:
                        st.warning("Mẫu gốc chưa có ảnh.")
