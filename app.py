import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CẤU HÌNH HỆ THỐNG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("Lỗi kết nối Supabase. Kiểm tra URL/KEY.")

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V5", page_icon="📏")

@st.cache_resource
def load_ai_model():
    # Sử dụng ResNet18 để nhận diện hình ảnh bản vẽ
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai_model()

# --- HÀM XỬ LÝ TOÁN HỌC & CHUẨN HÓA ---
def parse_measurement(text):
    """Chuyển đổi phân số (1 1/2, 23 3/4) thành số thập phân (1.5, 23.75)"""
    try:
        txt = str(text).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null', '0']: return 0
        
        # Regex tìm số nguyên, thập phân hoặc phân số
        matches = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not matches: return 0
        val_str = matches[0]
        
        if ' ' in val_str: # Trường hợp hỗn số: "1 1/2"
            parts = val_str.split()
            return float(parts[0]) + eval(parts[1])
        if '/' in val_str: # Trường hợp phân số: "3/4"
            return eval(val_str)
        return float(val_str)
    except:
        return 0

def clean_text(t):
    """Làm sạch tên hạng mục để so khớp chính xác hơn"""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- HÀM TRÍCH XUẤT ĐA NĂNG (CHO CẢ REITMANS & DÒNG HÀNG MỚI) ---
def extract_specs_from_pdf(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        # 1. Chụp ảnh trang 1 làm mẫu đối soát hình ảnh
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
        doc.close()

        # 2. Đọc bảng thông số
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm Sample Size (thường nằm ở header: Sample Size = 32)
                    current_sample_size = "32" # Mặc định theo ảnh bạn gửi
                    all_cell_text = " ".join([str(x) for x in df.values.flatten() if x])
                    size_match = re.search(r"SAMPLE SIZE\s*[:\-]*\s*(\d+)", all_cell_text, re.I)
                    if size_match: current_sample_size = size_match.group(1)

                    # Tìm vị trí cột Description và cột giá trị (NEW hoặc số Size)
                    desc_col, val_col = -1, -1
                    for r_idx, row in df.iterrows():
                        row_clean = [str(c).upper().strip() for c in row if c]
                        
                        # Điều kiện tìm cột
                        if "DESCRIPTION" in row_clean or "POM NAME" in row_clean:
                            desc_col = next((i for i, v in enumerate(row_clean) if "DESCRIPTION" in v or "POM NAME" in v), -1)
                            # Tìm cột giá trị: Ưu tiên cột trùng Sample Size, sau đó tới "NEW", "SPEC", "SAMPLE"
                            val_targets = [current_sample_size, "NEW", "SAMPLE", "SPEC", "32"]
                            for target in val_targets:
                                for i, v in enumerate(row_clean):
                                    if target == v or target in v:
                                        val_col = i; break
                                if val_col != -1: break
                            
                            # Lấy dữ liệu các dòng bên dưới header
                            if desc_col != -1 and val_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    p_name = str(d_row[desc_col]).strip().upper()
                                    if len(p_name) < 3 or any(x in p_name for x in ["TOL", "REF", "TOTAL"]): continue
                                    
                                    measurement = parse_measurement(d_row[val_col])
                                    if measurement > 0:
                                        full_specs[p_name] = measurement
                                break
        return {"specs": full_specs, "img": img_bytes}
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None

# --- GIAO DIỆN STREAMLIT ---
st.title("🔍 AI Fashion Auditor Pro V5")
st.info("Hệ thống tự động nhận diện thông số từ PDF (Reitmans & Generic Brands)")

# --- SIDEBAR: KHO DỮ LIỆU ---
with st.sidebar:
    st.header("📂 THƯ VIỆN MẪU")
    if st.button("🔄 Cập nhật số lượng"): st.rerun()
    
    files = st.file_uploader("Nạp Techpack Gốc", accept_multiple_files=True)
    if files and st.button("🚀 LƯU VÀO KHO"):
        p = st.progress(0)
        for i, f in enumerate(files):
            d = extract_specs_from_pdf(f)
            if d and d['specs']:
                # Tạo vector ảnh
                img = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
                
                # Lưu vào Supabase
                f_path = f"lib_{f.name.split('.')[0]}.png"
                supabase.storage.from_(BUCKET).upload(path=f_path, file=d['img'], file_options={"x-upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(f_path)
                }).execute()
            p.progress((i + 1) / len(files))
        st.success("Đã nạp mẫu thành công!")

# --- PHẦN CHÍNH: ĐỐI SOÁT ---
t_file = st.file_uploader("Tải file CẦN KIỂM TRA (Audit)", type="pdf")
if t_file:
    target = extract_specs_from_pdf(t_file)
    if target and target['specs']:
        st.success(f"Đã tìm thấy {len(target['specs'])} hạng mục thông số.")
        
        # Tìm mẫu trong kho
        db = supabase.table("ai_data").select("*").execute()
        if db.data:
            # So sánh AI Image
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for item in db.data:
                v_ref = np.array(item['vector']).reshape(1, -1)
                sim = float(cosine_similarity(v_test, v_ref)) * 100
                matches.append({"data": item, "sim": sim})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            
            # Hiển thị kết quả
            st.subheader(f"✅ Mẫu gốc khớp nhất: {best['data']['file_name']} ({best['sim']:.1f}%)")
            c1, c2 = st.columns(2)
            with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
            with c2: st.image(best['data']['image_url'], caption="Mẫu trong kho")

            # So sánh chi tiết từng POM
            diff_data = []
            ref_specs = best['data']['spec_json']
            ref_map = {clean_text(k): v for k, v in ref_specs.items()}

            for p_name, v_target in target['specs'].items():
                p_clean = clean_text(p_name)
                v_ref = ref_map.get(p_clean, 0)
                diff = round(v_target - v_ref, 3)
                
                diff_data.append({
                    "Hạng mục": p_name,
                    "Giá trị Kiểm tra": v_target,
                    "Giá trị Gốc": v_ref,
                    "Lệch": diff,
                    "Tình trạng": "🚩 Lệch" if abs(diff) > 0.01 else "✔️ Khớp"
                })

            df_res = pd.DataFrame(diff_data)
            st.table(df_res.style.applymap(lambda x: 'color: red' if x == "🚩 Lệch" else '', subset=['Tình trạng']))
            
            # Xuất Excel
            out = io.BytesIO()
            df_res.to_excel(out, index=False)
            st.download_button("📥 Tải báo cáo Audit", out.getvalue(), "Audit_Report.xlsx")
