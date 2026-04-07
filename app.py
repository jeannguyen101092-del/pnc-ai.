import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"            
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Kết nối Database thất bại: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V32", page_icon="📊")

# ================= 1. AI VISION & UTILS =================
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

def to_excel(df_details, df_pom, file_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_details.to_sheet(writer, sheet_name='Chi_Tiet_Thiet_Ke', index=False)
        df_pom.to_sheet(writer, sheet_name='Thong_So_POM', index=False)
    return output.getvalue()

# ================= 2. TRÍCH XUẤT ĐA DẠNG (QUẦN, ÁO, VÁY, ĐẦM) =================
def deep_detail_inspection(text):
    t = str(text).upper()
    details = {}
    
    # Phân loại SP
    if any(x in t for x in ['JACKET', 'VEST', 'COAT']): details['Loại'] = "🧥 Áo Khoác/Vest"
    elif any(x in t for x in ['DRESS', 'GOWN']): details['Loại'] = "👗 Đầm (Dress)"
    elif 'SKIRT' in t: details['Loại'] = "👗 Váy (Skirt)"
    elif any(x in t for x in ['SHIRT', 'TOP', 'TEE', 'POLO']): details['Loại'] = "👕 Áo Sơ mi/Thun"
    else: details['Loại'] = "👖 Quần/Khác"

    # Chi tiết túi & phụ liệu
    pockets = []
    if 'CHEST POCKET' in t: pockets.append("Túi Ngực")
    if 'CARGO' in t: pockets.append("Túi Hộp")
    if 'WELT' in t: pockets.append("Túi Mổ")
    details['Túi'] = ", ".join(pockets) if pockets else "Tiêu chuẩn"
    
    details['Phụ liệu'] = "Dây kéo (Zipper)" if 'ZIPPER' in t else "Nút/Chun"
    return details

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
                        key = cells[0].upper()
                        if any(k in key for k in pom_keys):
                            val = re.findall(r"\d+\.?\d*", "".join(cells[1:]))
                            if val: specs[key] = val[0]
        
        return {"spec": specs, "img": img, "details": deep_detail_inspection(raw_text)}
    except: return None

# ================= 3. UI CHÍNH =================
st.title("📊 AI FASHION AUDITOR - REPORTING V32")

# Lấy dữ liệu kho
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res else []
except: samples = []

with st.sidebar:
    st.header("⚙️ Cấu hình & Kho")
    st.info(f"Đang có {len(samples)} mẫu chuẩn trong kho.")
    up_pdfs = st.file_uploader("Nạp mẫu mới (Bulk)", type=['pdf'], accept_multiple_files=True)
    # (Phần code nạp dữ liệu giữ nguyên như cũ)

# Giao diện chọn mã so sánh
sample_list = ["--- TỰ ĐỘNG TÌM KIẾM (AI) ---"] + [s['file_name'] for s in samples]
selected_code = st.selectbox("🎯 Chọn mã hàng đối chứng:", sample_list)

test_files = st.file_uploader("📂 Tải Techpack kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_techpack(t_file)
        if not data_test: continue

        with st.expander(f"📄 Kết quả: {t_file.name}", expanded=True):
            best_match = None
            sim_score = 0.0
            
            # Logic so sánh
            v_test = get_vector(data_test['img'])
            if selected_code == "--- TỰ ĐỘNG TÌM KIẾM (AI) ---":
                if samples and v_test:
                    scores = [float(cosine_similarity([v_test], [s['vector']])) for s in samples]
                    idx = np.argmax(scores)
                    best_match, sim_score = samples[idx], scores[idx]
            else:
                best_match = next((s for s in samples if s['file_name'] == selected_code), None)
                if best_match and v_test:
                    sim_score = float(cosine_similarity([v_test], [best_match['vector']]))

            if best_match:
                # Hiển thị Score
                color = "green" if sim_score > 0.8 else "orange"
                st.markdown(f"### Độ tương đồng dáng: <span style='color:{color}'>{sim_score:.1%}</span>", unsafe_allow_html=True)
                
                # Chuẩn bị dữ liệu DataFrame
                df_det = pd.DataFrame([
                    {"Hạng mục": k, "File Test": v, "Mẫu Gốc": best_match['details'].get(k, "N/A"), 
                     "Kết quả": "✅ Khớp" if v == best_match['details'].get(k) else "❌ Khác"}
                    for k, v in data_test['details'].items()
                ])

                df_pom = []
                p_t, p_g = data_test['spec'], best_match.get('spec_json', {})
                for k in set(list(p_t.keys()) + list(p_g.keys())):
                    v_t, v_g = p_t.get(k, "-"), p_g.get(k, "-")
                    df_pom.append({"Thông số (POM)": k, "Test": v_t, "Gốc": v_g, "Lệch": "✅" if v_t == v_g else "❌"})
                df_pom = pd.DataFrame(df_pom)

                # Nút Xuất Excel
                excel_data = to_excel(df_det, df_pom, t_file.name)
                st.download_button(
                    label="📥 Tải Báo Cáo Đối Chiếu (.xlsx)",
                    data=excel_data,
                    file_name=f"SoSanh_{t_file.name.replace('.pdf','')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Hiển thị bảng trên Web
                col1, col2 = st.columns(2)
                with col1: st.table(df_det)
                with col2: st.table(df_pom)
                st.image([data_test['img'], best_match['img_url']], caption=["Bản Test", f"Gốc: {best_match['file_name']}"], width=350)
