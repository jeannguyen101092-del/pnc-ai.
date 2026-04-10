import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
# Vui lòng điền thông tin Supabase của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V43.8", page_icon="📊")

# --- LOAD MODEL AI ---
@st.cache_resource
def load_ai():
    # Sử dụng ResNet18 để lấy vector đặc trưng của hình ảnh
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- UTILS: XỬ LÝ SỐ & CHUẨN HÓA TÊN ---
def parse_val(t):
    """Xử lý các định dạng số: 1 1/2, 3/4, 10.5, 10"""
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null']: return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def normalize_pom(t):
    """Chuẩn hóa tên vị trí đo để so khớp dễ hơn"""
    if not t: return ""
    # Viết hoa, bỏ dấu câu, bỏ khoảng trắng thừa
    s = str(t).upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    return " ".join(s.split())

# --- TRÍCH XUẤT DỮ LIỆU ĐA CỘT (MULTI-SIZE) ---
def extract_tp_data(pdf_file):
    img_bytes = None
    all_tables = []
    pdf_content = pdf_file.read()
    
    # 1. Lấy ảnh đại diện
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Quét bảng thông số
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tbs = page.extract_tables()
            for tb in tbs:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                
                # Làm sạch Header (Lấy dòng đầu tiên có chữ làm tiêu đề)
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True).fillna("")
                all_tables.append(df)
                
    return {"img": img_bytes, "tables": all_tables}

def get_image_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
        return vec
    except: return None

# --- GIAO DIỆN SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📦 KHO DỮ LIỆU MẪU")
    uploaded_files = st.file_uploader("Nạp Techpack chuẩn", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 LƯU VÀO HỆ THỐNG"):
        for f in uploaded_files:
            data = extract_tp_data(f)
            if data['tables']:
                # Lấy bảng dài nhất làm chuẩn
                main_df = max(data['tables'], key=len)
                vec = get_image_vector(data['img'])
                
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                res_url = supabase.storage.from_(BUCKET).get_public_url(path)
                img_url = res_url if isinstance(res_url, str) else getattr(res_url, "public_url", "")

                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": main_df.to_dict('records'),
                    "image_url": img_url
                }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- GIAO DIỆN CHÍNH: AUDIT ---
st.title("🔍 AI Fashion Auditor V43.8 PRO")

f_target = st.file_uploader("📤 Upload bản vẽ cần đối soát", type="pdf")

if f_target:
    target_data = extract_tp_data(f_target)
    
    if target_data['tables']:
        # Lấy bảng đầu tiên của file upload
        df_target = target_data['tables'][0]
        cols = df_target.columns.tolist()
        
        # --- BƯỚC 1: CHỌN VỊ TRÍ & SIZE ---
        st.write("### ⚙️ Thiết lập đối soát")
        c1, c2 = st.columns(2)
        with c1:
            pom_col = st.selectbox("Cột chứa Tên vị trí (Description/POM):", cols, index=0)
        with c2:
            size_col = st.selectbox("Chọn cột Size cần Audit (S, M, L...):", [c for c in cols if c != pom_col])

        # --- BƯỚC 2: TÌM MẪU KHỚP BẰNG AI ---
        v_test = get_image_vector(target_data['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        if db.data:
            matches = []
            for row in db.data:
                sim = cosine_similarity([v_test], [row['vector']])[0][0] if v_test and row['vector'] else 0
                matches.append({"row": row, "sim": sim * 100})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            ref_row = best['row']
            df_ref = pd.DataFrame(ref_row['spec_json'])

            st.divider()
            st.subheader(f"✨ Kết quả: Khớp với mẫu `{ref_row['file_name']}` ({best['sim']:.1f}%)")

            # Hiển thị ảnh so sánh
            im1, im2 = st.columns(2)
            with im1: st.image(target_data['img'], caption="Bản vẽ Kiểm tra")
            with im2: st.image(ref_row['image_url'], caption="Mẫu gốc hệ thống")

            # --- BƯỚC 3: SO SÁNH ĐÚNG DÒNG (LOGIC QUAN TRỌNG) ---
            st.write(f"### 📊 Bảng đối chiếu chi tiết (Size: {size_col})")
            
            # Tạo bản đồ Map từ file mẫu (Nomalize POM -> Value)
            ref_map = {}
            # Giả định cột đầu tiên của bảng mẫu là tên POM
            ref_pom_col_name = df_ref.columns[0] 
            for _, r in df_ref.iterrows():
                key = normalize_pom(r[ref_pom_col_name])
                # Lấy giá trị cột size tương ứng, nếu không có lấy cột đầu tiên sau POM
                val = parse_val(r.get(size_col, r.iloc[1]))
                ref_map[key] = val

            # Xây dựng bảng so sánh dựa trên file upload
            comparison = []
            for _, r_t in df_target.iterrows():
                raw_name = r_t[pom_col]
                norm_name = normalize_pom(raw_name)
                if not norm_name: continue
                
                val_actual = parse_val(r_t[size_col])
                val_standard = ref_map.get(norm_name, 0)
                diff = round(val_actual - val_standard, 3)

                comparison.append({
                    "Vị trí đo (Description)": raw_name,
                    "Thực tế (Kiểm)": val_actual,
                    "Tiêu chuẩn (Mẫu)": val_standard,
                    "Chênh lệch": diff,
                    "Kết luận": "✅ Đạt" if abs(diff) <= 0.125 else "❌ Lệch"
                })

            res_df = pd.DataFrame(comparison)
            
            # Hiển thị bảng màu sắc
            def color_diff(row):
                return ['background-color: #ffcccc' if row['Chênh lệch'] != 0 else '' for _ in row]

            st.dataframe(res_df.style.apply(color_diff, axis=1), use_container_width=True, height=600)

            # Xuất Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False, sheet_name='Audit_Report')
            st.download_button("📥 Tải báo cáo Audit", output.getvalue(), f"Audit_{ref_row['file_name']}.xlsx")

st.divider()
if st.button("♻️ Reset Hệ thống"):
    st.rerun()
