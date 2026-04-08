import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. CẤU HÌNH & KẾT NỐI =================
# Thay thế URL và KEY của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Kết nối Database thất bại: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V33", page_icon="📊")

# ================= 2. HỆ THỐNG AI VISION =================
@st.cache_resource
def load_vision_ai():
    # Sử dụng ResNet50 để tạo vector đặc trưng cho hình ảnh thiết kế
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
    except Exception: return None

def to_excel(df_details, df_pom):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_details.to_excel(writer, sheet_name='Chi_Tiet_Thiet_Ke', index=False)
        df_pom.to_excel(writer, sheet_name='Thong_So_POM', index=False)
    return output.getvalue()

# ================= 3. TRÍCH XUẤT DỮ LIỆU TECHPACK =================
def deep_detail_inspection(text):
    t = str(text).upper()
    det = {}
    # Phân loại sản phẩm
    if any(x in t for x in ['JACKET', 'VEST', 'COAT', 'BLAZER']): det['Loại'] = "🧥 Áo Khoác/Vest"
    elif any(x in t for x in ['DRESS', 'GOWN']): det['Loại'] = "👗 Đầm (Dress)"
    elif 'SKIRT' in t: det['Loại'] = "👗 Váy (Skirt)"
    elif any(x in t for x in ['SHIRT', 'TOP', 'TEE', 'POLO']): det['Loại'] = "👕 Áo Sơ mi/Thun"
    elif any(x in t for x in ['PANT', 'TROUSER', 'SHORT', 'JEAN']): det['Loại'] = "👖 Quần"
    else: det['Loại'] = "📦 Khác"

    # Trích xuất túi & phụ liệu
    pockets = []
    if 'CHEST POCKET' in t: pockets.append("Túi Ngực")
    if 'CARGO' in t: pockets.append("Túi Hộp")
    if 'WELT' in t: pockets.append("Túi Mổ")
    det['Túi'] = ", ".join(pockets) if pockets else "Tiêu chuẩn"
    det['Phụ liệu'] = "Dây kéo (Zipper)" if 'ZIPPER' in t else "Nút/Chun"
    return det

def extract_techpack(pdf_file):
    specs, img, raw_text = {}, None, ""
    pom_keys = ['WAIST', 'HIP', 'CHEST', 'BUST', 'LENGTH', 'SHOULDER', 'SLEEVE', 'RISE', 'THIGH', 'LEG']
    try:
        pdf_bytes = pdf_file.read()
        # 1. Trích xuất hình ảnh trang đầu tiên
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        
        # 2. Trích xuất bảng và văn bản
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "")
                tables = page.extract_tables()
                for tb in tables:
                    for row in tb:
                        cells = [str(c).strip() for c in row if c]
                        if len(cells) < 2: continue
                        k_upper = str(cells[0]).upper()
                        if any(k in k_upper for k in pom_keys):
                            val = re.findall(r"\d+\.?\d*", " ".join(cells[1:]))
                            if val: specs[k_upper] = val[0]
        return {"spec": specs, "img": img, "details": deep_detail_inspection(raw_text)}
    except Exception: return None

# ================= 4. GIAO DIỆN CHÍNH =================
st.title("📊 AI FASHION AUDITOR V33 - FULL AUDIT")

# Tải danh sách mẫu từ Database
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res else []
except Exception: samples = []

with st.sidebar:
    st.header("⚙️ Cấu hình kho")
    st.metric("MẪU TRONG KHO", len(samples))
    up_pdfs = st.file_uploader("Nạp PDF vào kho (Gốc)", type=['pdf'], accept_multiple_files=True)
    if up_pdfs and st.button("🚀 NẠP DỮ LIỆU"):
        for f in up_pdfs:
            d = extract_techpack(f)
            if d:
                ma = f.name.replace(".pdf", "")
                vec = get_vector(d['img'])
                supabase.table("ai_data").upsert({
                    "file_name": ma, 
                    "vector": vec, 
                    "spec_json": d['spec'], 
                    "details": d['details']
                }).execute()
        st.success("Đã cập nhật kho dữ liệu!")
        st.rerun()

# Khu vực đối chiếu
sample_list = ["--- TỰ ĐỘNG TÌM KIẾM (AI) ---"] + [s['file_name'] for s in samples]
selected_code = st.selectbox("🎯 1. Chọn mã đối chứng:", sample_list)
test_files = st.file_uploader("📂 2. Tải Techpack cần kiểm tra", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_techpack(t_file)
        if not data_test: 
            st.error(f"Không thể đọc file {t_file.name}")
            continue

        with st.expander(f"📑 KẾT QUẢ ĐỐI CHIẾU: {t_file.name}", expanded=True):
            best_match, sim_score = None, 0.0
            v_test = get_vector(data_test['img'])
            
            # Logic Tìm kiếm mã khớp
            if v_test:
                vt_np = np.array(v_test).reshape(1, -1)
                if selected_code == "--- TỰ ĐỘNG TÌM KIẾM (AI) ---":
                    scores, valid_candidates = [], []
                    for s in samples:
                        try:
                            v_raw = s.get('vector')
                            if isinstance(v_raw, str): v_raw = json.loads(v_raw)
                            if v_raw:
                                score = float(cosine_similarity(vt_np, np.array(v_raw).reshape(1, -1)))
                                scores.append(score)
                                valid_candidates.append(s)
                        except: continue
                    if scores:
                        idx = np.argmax(scores)
                        best_match, sim_score = valid_candidates[idx], scores[idx]
                else:
                    best_match = next((s for s in samples if s['file_name'] == selected_code), None)
                    if best_match:
                        try:
                            v_raw = best_match.get('vector')
                            if isinstance(v_raw, str): v_raw = json.loads(v_raw)
                            sim_score = float(cosine_similarity(vt_np, np.array(v_raw).reshape(1, -1)))
                        except: sim_score = 0.0

            # HIỂN THỊ KẾT QUẢ SO SÁNH
            if best_match:
                st.success(f"✅ Đã khớp với mã: **{best_match['file_name']}** | Độ tương đồng dáng: **{sim_score:.1%}**")
                
                # Ép kiểu dữ liệu an toàn
                goc_details = best_match.get('details', {})
                if isinstance(goc_details, str): goc_details = json.loads(goc_details)
                goc_spec = best_match.get('spec_json', {})
                if isinstance(goc_spec, str): goc_spec = json.loads(goc_spec)
                
                # Bảng 1: Chi tiết thiết kế
                st.subheader("🔍 1. Chi tiết thiết kế")
                rows_det = []
                for k, v in data_test['details'].items():
                    v_goc = goc_details.get(k, "N/A")
                    rows_det.append({
                        "Hạng mục": k, "Bản Test": v, "Bản Gốc": v_goc, 
                        "Kết quả": "✅ Khớp" if str(v) == str(v_goc) else "❌ Khác"
                    })
                df_det = pd.DataFrame(rows_det)
                
                # Bảng 2: Thông số POM
                st.subheader("📏 2. Thông số kỹ thuật (POM)")
                all_keys = sorted(set(list(data_test['spec'].keys()) + list(goc_spec.keys())))
                rows_pom = []
                for k in all_keys:
                    v_t = data_test['spec'].get(k, "-")
                    v_g = goc_spec.get(k, "-")
                    rows_pom.append({
                        "Thông số": k, "Test": v_t, "Gốc": v_g, 
                        "Lệch": "✅" if str(v_t) == str(v_g) else "❌"
                    })
                df_pom = pd.DataFrame(rows_pom)

                st.download_button("📥 Tải Báo Cáo Excel", data=to_excel(df_det, df_pom), file_name=f"SoSanh_{best_match['file_name']}.xlsx")
                
                c1, c2 = st.columns(2)
                with c1: st.dataframe(df_det, use_container_width=True)
                with c2: st.dataframe(df_pom, use_container_width=True)
                
                # Hiển thị hình ảnh
                st.divider()
                st.subheader("🖼️ Hình ảnh đối chứng")
                col_img_test, col_img_goc = st.columns(2)
                with col_img_test:
                    st.image(data_test['img'], caption="Hình từ File đang kiểm tra", use_container_width=True)
                with col_img_goc:
                    st.info("💡 Hình ảnh gốc nằm trong hồ sơ lưu trữ hệ thống.")
            else:
                st.warning("Không tìm thấy mẫu đối chứng phù hợp trong kho.")
