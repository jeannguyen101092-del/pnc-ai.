import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
# Thay đổi thông tin kết nối Supabase của bạn tại đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V95", page_icon="📏")

# Khởi tạo trạng thái để reset file uploader
if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .status-khop { color: #28a745; font-weight: bold; }
    .status-lech { color: #dc3545; font-weight: bold; }
    thead th { background-color: #f0f2f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI & CÔNG CỤ HỖ TRỢ =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    
    # Danh sách từ khóa đặc trưng
    keywords = {
        "VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM", "GOWN"],
        "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM", "QUẦN"],
        "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "COAT", "ÁO", "SWEATER"]
    }
    
    # Đếm số lần xuất hiện của từng loại
    scores = {"VÁY/ĐẦM": 0, "QUẦN": 0, "ÁO": 0}
    for cat, keys in keywords.items():
        for k in keys:
            scores[cat] += t.count(k)
    
    # Lấy loại có điểm cao nhất
    detected = max(scores, key=scores.get)
    
    # Nếu không có từ khóa nào, mặc định là KHÁC
    return detected if scores[detected] > 0 else "KHÁC"

def parse_val(t):
    try:
        # Nếu là chuỗi quá dài hoặc chứa ký tự đặc biệt của ID thì bỏ qua
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 10 or not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs", "date", "2024", "2025"]): 
            return 0
            
        # Regex tìm số đo (bao gồm cả phân số)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        
        v_str = match[0]
        if ' ' in v_str:
            parts = v_str.split()
            val = float(parts[0]) + eval(parts[1])
        else:
            val = eval(v_str) if '/' in v_str else float(v_str)
            
        # LỌC NHIỄU: Thông số may mặc thực tế thường nằm trong khoảng 0.1 đến 150. 
        # Nếu con số trích xuất ra là 10001337 hay 313.000 (do lỗi đọc text dính chùm) thì bỏ qua.
        if val > 200: return 0 
        return val
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
    return vec.astype(float).tolist()

# ================= 3. TRÍCH XUẤT THÔNG MINH (DÒ CỘT SỐ) =================
def extract_pdf_v95(file):
    specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text_list.append(str(page.get_text() or ""))
        doc.close()
        
        full_text = " ".join(full_text_list)
        category = detect_category(full_text, file.name)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER", "THIGH", "RISE"] if k in flat_text) < 2: continue

                    n_col, v_col = -1, -1
                    # Tìm cột tên POM
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): n_col = i; break
                        if n_col != -1: break
                    
                    # Tìm cột chứa nhiều số đo nhất
                    if n_col != -1:
                        max_nums = 0
                        for i in range(len(df.columns)):
                            if i == n_col: continue
                            num_count = sum(1 for val in df.iloc[:12, i] if parse_val(val) > 0)
                            if num_count > max_nums: max_nums = num_count; v_col = i

                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if len(name) > 3 and val > 0 and not any(x in name for x in ["DESCRIPTION", "POM"]):
                                specs[name] = val
                if specs: break 
        return {"specs": specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR (QUẢN LÝ KHO & TỰ XÓA FILE) =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ MẪU")
    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng số mẫu hiện có", f"{res_db.count or 0} file")
    except: st.error("Lỗi kết nối database.")

    st.divider()
    # Sử dụng up_key để reset uploader sau khi nạp xong
    new_files = st.file_uploader("Nạp Techpack mới vào kho", accept_multiple_files=True, key=f"uploader_{st.session_state.up_key}")
    
    if new_files and st.button("🚀 XÁC NHẬN NẠP", use_container_width=True):
        for f in new_files:
            with st.spinner(f"Đang nạp {f.name}..."):
                data = extract_pdf_v95(f)
                if data and data['specs']:
                    vec = get_image_vector(data['img'])
                    # Làm sạch tên file để lưu trữ
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))
                    path = f"lib_{clean_name}.png"
                    supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(path)
                    supabase.table("ai_data").insert({
                        "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                        "image_url": url, "category": data['category']
                    }).execute()
        
        st.success("Nạp thành công!")
        st.session_state.up_key += 1 # Thay đổi key để xóa danh sách file đã chọn
        st.rerun()

# ================= 5. LUỒNG ĐỐI SOÁT CHÍNH =================
st.title("🔍 AI SMART AUDITOR - V95")

file_audit = st.file_uploader("📤 Upload file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    with st.spinner("Đang trích xuất và đối chiếu..."):
        target = extract_pdf_v95(file_audit)
    
    if target and target["specs"]:
        st.info(f"✨ Phát hiện: **{target['category']}** | {len(target['specs'])} vị trí đo.")
        
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        if res.data:
            # Tìm kiếm độ tương đồng ảnh
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            matches = []
            for item in res.data:
                sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))[0][0]
                matches.append({**item, "sim": sim})
            
            # Lấy Top 3 mã tương đồng nhất
            top_3 = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]
            
            st.subheader("🖼️ TOP 3 MẪU TƯƠNG ĐỒNG TRONG KHO")
            cols = st.columns(3)
            for i, m in enumerate(top_3):
                with cols[i]:
                    st.image(m['image_url'], caption=f"Top {i+1}: {m['file_name']} ({m['sim']:.1%})")

            # Người dùng chọn 1 trong 3 mẫu để xem bảng chi tiết
            selected_name = st.selectbox("Chọn mẫu gốc để so sánh thông số chi tiết:", [m['file_name'] for m in top_3])
            best_match = next(m for m in top_3 if m['file_name'] == selected_name)

            # --- BẢNG ĐỐI SOÁT ---
            st.subheader(f"📊 BẢNG SO SÁNH: {selected_name}")
            audit_rows = []
            for pom, val in target['specs'].items():
                master_val = best_match['spec_json'].get(pom, 0)
                diff = round(val - master_val, 3) if master_val else 0
                
                status = "✅ Khớp"
                if master_val == 0: status = "❓ Không tìm thấy"
                elif abs(diff) > 0.125: status = f"❌ Lệch ({diff})"
                
                audit_rows.append({"Vị trí đo (POM)": pom, "File Mới": val, "Mẫu Gốc": master_val, "Kết quả": status})
            
            df_audit = pd.DataFrame(audit_rows)
            st.table(df_audit)

            # --- XUẤT EXCEL ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_audit.to_excel(writer, index=False, sheet_name='Audit_Report')
                # Tự động căn chỉnh độ rộng cột Excel
                worksheet = writer.sheets['Audit_Report']
                for idx, col in enumerate(df_audit.columns):
                    worksheet.set_column(idx, idx, 20)
            
            st.download_button(
                label="📥 TẢI BÁO CÁO EXCEL",
                data=output.getvalue(),
                file_name=f"Audit_Report_{selected_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning(f"Chưa có mẫu nào thuộc loại **{target['category']}** trong kho để đối chiếu.")
    else:
        st.error("Không thể đọc được bảng thông số. Vui lòng kiểm tra lại định dạng PDF.")

st.divider()
st.caption("AI Smart Auditor V95 - Chuyên nghiệp cho ngành may mặc.")
