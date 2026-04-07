import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. CẤU HÌNH HỆ THỐNG =================
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
        df_details.to_excel(writer, sheet_name='Chi_Tiet_Thiet_Ke', index=False)
        df_pom.to_excel(writer, sheet_name='Thong_So_POM', index=False)
    return output.getvalue()

# ================= 3. TRÍCH XUẤT ĐA DẠNG (ÁO, QUẦN, VÁY, ĐẦM) =================
def deep_detail_inspection(text):
    t = str(text).upper()
    det = {}
    
    # Phân loại sản phẩm
    if any(x in t for x in ['JACKET', 'VEST', 'COAT']): det['Loại'] = "🧥 Áo Khoác/Vest"
    elif any(x in t for x in ['DRESS', 'GOWN']): det['Loại'] = "👗 Đầm (Dress)"
    elif 'SKIRT' in t: det['Loại'] = "👗 Váy (Skirt)"
    elif any(x in t for x in ['SHIRT', 'TOP', 'TEE', 'POLO']): det['Loại'] = "👕 Áo Sơ mi/Thun"
    elif any(x in t for x in ['PANT', 'TROUSER', 'SHORT']): det['Loại'] = "👖 Quần"
    else: det['Loại'] = "📦 Khác"

    # Chi tiết thiết kế
    pockets = []
    if 'CHEST POCKET' in t: pockets.append("Túi Ngực")
    if 'CARGO' in t: pockets.append("Túi Hộp")
    if 'WELT' in t: pockets.append("Túi Mổ")
    det['Túi'] = ", ".join(pockets) if pockets else "Tiêu chuẩn"
    det['Phụ liệu'] = "Dây kéo" if 'ZIPPER' in t else "Nút/Chun"
    
    return det

def extract_techpack(pdf_file):
    specs, img, raw_text = {}, None, ""
    pom_keys = ['WAIST', 'HIP', 'CHEST', 'BUST', 'LENGTH', 'SHOULDER', 'SLEEVE', 'RISE', 'THIGH', 'LEG']
    
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "")
                for tb in page.extract_tables():
                    for row in tb:
                        cells = [str(c).strip() for c in row if c]
                        if len(cells) < 2: continue
                        k_upper = cells[0].upper()
                        if any(k in k_upper for k in pom_keys):
                            val = re.findall(r"\d+\.?\d*", " ".join(cells[1:]))
                            if val: specs[k_upper] = val[0]
        
        return {"spec": specs, "img": img, "details": deep_detail_inspection(raw_text)}
    except: return None

# ================= 4. GIAO DIỆN CHÍNH =================
st.title("📊 AI FASHION AUDITOR V33 - FULL AUDIT")

try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res else []
except: samples = []

with st.sidebar:
    st.header("⚙️ Cấu hình kho")
    st.metric("MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF gốc (Bulk)", type=['pdf'], accept_multiple_files=True)
    if up_pdfs and st.button("🚀 NẠP DỮ LIỆU"):
        for f in up_pdfs:
            d = extract_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                supabase.table("ai_data").upsert({
                    "file_name": ma, "vector": vec, "spec_json": d['spec'], "details": d['details']
                }).execute()
        st.rerun()

# Chọn mã hàng
sample_list = ["--- TỰ ĐỘNG TÌM KIẾM (AI) ---"] + [s['file_name'] for s in samples]
selected_code = st.selectbox("🎯 Bước 1: Chọn mã hàng đối chứng:", sample_list)

test_files = st.file_uploader("📂 Bước 2: Tải Techpack kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_techpack(t_file)
        if not data_test: continue

        with st.expander(f"📑 ĐỐI CHIẾU: {t_file.name}", expanded=True):
            best_match, sim_score = None, 0.0
            v_test = get_vector(data_test['img'])
            
            # --- XỬ LÝ SO SÁNH (SỬA LỖI TYPEERROR) ---
            if selected_code == "--- TỰ ĐỘNG TÌM KIẾM (AI) ---":
                valid_samples = [s for s in samples if s.get('vector')]
                if valid_samples and v_test:
                    vt_np = np.array(v_test).reshape(1, -1)
                    scores = [float(cosine_similarity(vt_np, np.array(s['vector']).reshape(1, -1))) for s in valid_samples]
                    idx = np.argmax(scores)
                    best_match, sim_score = valid_samples[idx], scores[idx]
            else:
                best_match = next((s for s in samples if s['file_name'] == selected_code), None)
                if best_match and v_test and best_match.get('vector'):
                    vt_np = np.array(v_test).reshape(1, -1)
                    sim_score = float(cosine_similarity(vt_np, np.array(best_match['vector']).reshape(1, -1)))

            if best_match:
                st.markdown(f"### Độ tương đồng: `{sim_score:.1%}`")
                
                # Dữ liệu Excel & Hiển thị
                df_det = pd.DataFrame([{"Hạng mục": k, "Test": v, "Gốc": best_match['details'].get(k, "N/A"), "Kết quả": "✅" if v == best_match['details'].get(k) else "❌"} for k, v in data_test['details'].items()])
                
                p_t, p_g = data_test['spec'], best_match.get('spec_json', {})
                df_pom = pd.DataFrame([{"Thông số": k, "Test": p_t.get(k, "-"), "Gốc": p_g.get(k, "-"), "Lệch": "✅" if p_t.get(k) == p_g.get(k) else "❌"} for k in set(list(p_t.keys()) + list(p_g.keys()))])

                st.download_button("📥 Tải Báo Cáo Excel", data=to_excel(df_det, df_pom), file_name=f"Report_{best_match['file_name']}.xlsx")

                c1, c2 = st.columns(2)
                with c1: st.table(df_det)
                with c2: st.table(df_pom)
                st.image(data_test['img'], caption="Ảnh trích xuất", width=400)
            else:
                st.warning("Không tìm thấy dữ liệu đối xứng.")
