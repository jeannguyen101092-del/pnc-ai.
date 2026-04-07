import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay URL và KEY thực tế của bạn) =================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"            
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Kết nối Database thất bại: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V32", page_icon="📊")

# ================= 1. HỆ THỐNG AI VISION =================
@st.cache_resource
def load_vision_ai():
    # Sử dụng ResNet50 để trích xuất đặc trưng hình ảnh
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

# Hàm xuất Excel (Sửa lỗi ghi đè sheet)
def to_excel(df_details, df_pom):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_details.to_excel(writer, sheet_name='Chi_Tiet_Thiet_Ke', index=False)
        df_pom.to_excel(writer, sheet_name='Thong_So_POM', index=False)
    return output.getvalue()

# ================= 2. TRÍCH XUẤT ĐA DẠNG (QUẦN, ÁO, VÁY, ĐẦM) =================
def deep_detail_inspection(text):
    t = str(text).upper()
    details = {}
    
    # Phân loại sản phẩm thông minh
    if any(x in t for x in ['JACKET', 'VEST', 'COAT', 'BLAZER']): details['Loại'] = "🧥 Áo Khoác/Vest"
    elif any(x in t for x in ['DRESS', 'GOWN']): details['Loại'] = "👗 Đầm (Dress)"
    elif 'SKIRT' in t: details['Loại'] = "👗 Váy (Skirt)"
    elif any(x in t for x in ['SHIRT', 'TOP', 'TEE', 'POLO']): details['Loại'] = "👕 Áo Sơ mi/Thun"
    elif any(x in t for x in ['PANT', 'TROUSER', 'SHORT', 'JEAN']): details['Loại'] = "👖 Quần"
    else: details['Loại'] = "📦 Khác/Phụ kiện"

    # Soi chi tiết thiết kế
    pockets = []
    if 'CHEST POCKET' in t: pockets.append("Túi Ngực")
    if 'CARGO' in t: pockets.append("Túi Hộp")
    if 'WELT' in t: pockets.append("Túi Mổ")
    if 'SLANT' in t: pockets.append("Túi Xéo")
    details['Túi'] = ", ".join(pockets) if pockets else "Tiêu chuẩn"
    
    details['Phụ liệu'] = "Dây kéo (Zipper)" if 'ZIPPER' in t else "Cúc/Chun"
    if 'HOODED' in t: details['Nón'] = "Có nón"
    
    return details

def extract_techpack(pdf_file):
    specs, img, raw_text = {}, None, ""
    # Từ khóa POM cho cả áo và quần
    pom_keys = ['WAIST', 'HIP', 'CHEST', 'BUST', 'LENGTH', 'SHOULDER', 'SLEEVE', 'RISE', 'THIGH', 'LEG', 'OPENING']
    
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "")
                tables = page.extract_tables()
                for tb in tables:
                    for row in tb:
                        cells = [str(c).strip() for c in row if c]
                        if len(cells) < 2: continue
                        key = cells[0].upper()
                        if any(k in key for k in pom_keys):
                            # Tìm con số đầu tiên trong các cột còn lại
                            val_match = re.findall(r"\d+\.?\d*", " ".join(cells[1:]))
                            if val_match: specs[key] = val_match[0]
        
        return {"spec": specs, "img": img, "details": deep_detail_inspection(raw_text)}
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# ================= 3. GIAO DIỆN CHÍNH =================
st.title("📊 AI FASHION AUDITOR - REPORTING V32")

# Tải dữ liệu kho mẫu
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res else []
except: samples = []

with st.sidebar:
    st.header("⚙️ Cấu hình & Kho")
    st.info(f"Đang có {len(samples)} mẫu chuẩn trong kho.")
    up_pdfs = st.file_uploader("Nạp mẫu mới (Bulk)", type=['pdf'], accept_multiple_files=True)
    
    if up_pdfs and st.button("🚀 NẠP VÀO KHO"):
        for f in up_pdfs:
            d = extract_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                # Upload ảnh và lưu Data (Rút gọn cho nhanh)
                supabase.table("ai_data").upsert({
                    "file_name": ma, "vector": vec, "spec_json": d['spec'], 
                    "img_url": "", "details": d['details']
                }).execute()
                st.toast(f"Đã nạp: {ma}")
        st.rerun()

# --- LỰA CHỌN MÃ HÀNG ---
sample_list = ["--- TỰ ĐỘNG TÌM KIẾM (AI) ---"] + [s['file_name'] for s in samples]
selected_code = st.selectbox("🎯 Chọn mã hàng đối chứng để so sánh:", sample_list)

test_files = st.file_uploader("📂 Tải Techpack cần kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_techpack(t_file)
        if not data_test: continue

        with st.expander(f"📄 Đang phân tích: {t_file.name}", expanded=True):
            best_match = None
            sim_score = 0.0
            v_test = get_vector(data_test['img'])
            
            # --- LOGIC ĐỐI CHIẾU ---
            if selected_code == "--- TỰ ĐỘNG TÌM KIẾM (AI) ---":
                if samples and v_test:
                    # LỌC CÁC MẪU CÓ VECTOR HỢP LỆ ĐỂ TRÁNH LỖI TYPEERROR
                    valid_samples = [s for s in samples if s.get('vector') is not None]
                    if valid_samples:
                        # SỬA LỖI: Chuyển vector về numpy array 2D (1, -1)
                        vec_test_np = np.array(v_test).reshape(1, -1)
                        scores = [float(cosine_similarity(vec_test_np, np.array(s['vector']).reshape(1, -1))) for s in valid_samples]
                        idx = np.argmax(scores)
                        best_match, sim_score = valid_samples[idx], scores[idx]
            else:
                best_match = next((s for s in samples if s['file_name'] == selected_code), None)
                if best_match and v_test and best_match.get('vector'):
                    vec_test_np = np.array(v_test).reshape(1, -1)
                    sim_score = float(cosine_similarity(vec_test_np, np.array(best_match['vector']).reshape(1, -1)))

            # --- HIỂN THỊ KẾT QUẢ ---
            if best_match:
                st.markdown(f"### Độ tương đồng hình ảnh: `{sim_score:.1%}`")
                
                # Bảng chi tiết thiết kế
                df_det = pd.DataFrame([
                    {"Hạng mục": k, "File Test": v, "Mẫu Gốc": best_match['details'].get(k, "N/A"), 
                     "Kết quả": "✅ Khớp" if v == best_match['details'].get(k) else "❌ Khác"}
                    for k, v in data_test['details'].items()
                ])

                # Bảng thông số POM
                p_t, p_g = data_test['spec'], best_match.get('spec_json', {})
                df_pom_data = []
                for k in set(list(p_t.keys()) + list(p_g.keys())):
                    v_t, v_g = p_t.get(k, "-"), p_g.get(k, "-")
                    df_pom_data.append({"Thông số": k, "Test": v_t, "Gốc": v_g, "Sai lệch": "✅" if v_t == v_g else "❌"})
                df_pom = pd.DataFrame(df_pom_data)

                # Nút Xuất Excel
                excel_file = to_excel(df_det, df_pom)
                st.download_button(
                    label="📥 Tải Báo Cáo Đối Chiếu (.xlsx)",
                    data=excel_file,
                    file_name=f"Report_{t_file.name.replace('.pdf','')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Hiển thị trực quan
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🕵️ Chi tiết thiết kế")
                    st.table(df_det)
                with c2:
                    st.subheader("📏 Thông số POM")
                    st.table(df_pom)
                
                st.image(data_test['img'], caption="Hình ảnh trích xuất từ file Test", width=400)
            else:
                st.warning("⚠️ Không tìm thấy mẫu đối chứng phù hợp trong kho.")
