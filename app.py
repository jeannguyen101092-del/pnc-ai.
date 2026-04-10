import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Auditor V49 PRO", page_icon="🔍")

# ================= AI MODEL =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

model_ai = load_ai()

# ================= UTILS =================
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null']: return 0
        # Regex bắt: 1 1/2, 1/2, 1.5, 10
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v: # Xử lý hỗn số: "1 1/2" -> 1.5
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT MULTI-SIZE =================
def extract_pom_pro(pdf_file):
    brand = "OTHER"
    img_bytes = None
    pdf_content = pdf_file.read()
    
    # 1. Detect Brand & Get Image
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += (page.get_text() or "").upper() + " "
    if "REITMANS" in all_text: brand = "REITMANS"
    
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Extract Multi-column Tables (Sizes)
    table_data = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                # Làm sạch header
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:]
                table_data.append(df)

    return {"img": img_bytes, "tables": table_data, "brand": brand}

# ================= VECTOR =================
def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        with torch.no_grad():
            v = model_ai(tf(img).unsqueeze(0)).view(-1).numpy()
            v = v / np.linalg.norm(v)
        return v.tolist()
    except: return None

# ================= SIDEBAR: NẠP DỮ LIỆU MẪU =================
with st.sidebar:
    st.header("📂 HỆ THỐNG DỮ LIỆU")
    count_res = supabase.table("ai_data").select("*", count="exact").execute()
    st.metric("Tổng số mẫu", count_res.count if count_res.count else 0)
    
    upload_files = st.file_uploader("Nạp Techpack Mẫu", accept_multiple_files=True)
    if upload_files and st.button("🚀 LƯU VÀO DB"):
        for f in upload_files:
            data = extract_pom_pro(f)
            # Lưu bảng đầu tiên của file mẫu làm chuẩn (size-json)
            specs = data['tables'][0].to_dict('records') if data['tables'] else []
            vec = get_vector(data['img'])
            
            img_url = ""
            if data['img']:
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)["publicUrl"]

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": specs, # Lưu nguyên bảng list dict
                "image_url": img_url
            }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# ================= MAIN: AUDIT KIỂM TRA =================
st.title("🔥 AI AUDITOR V49 - SMART COMPARISON")

file_test = st.file_uploader("📤 Upload file PDF cần kiểm tra", type="pdf")

if file_test:
    target = extract_pom_pro(file_test)
    
    if not target['tables']:
        st.error("Không tìm thấy bảng thông số trong file PDF.")
    else:
        df_target = target['tables'][0] # Lấy bảng đầu tiên tìm thấy
        all_cols = df_target.columns.tolist()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info(f"Brand detected: **{target['brand']}**")
            # Chọn cột POM (Thường là Description hoặc POM)
            pom_col = st.selectbox("Chọn cột Vị trí (POM):", all_cols, index=0)
            # Chọn Size chuẩn để kiểm
            size_col = st.selectbox("Chọn Size chuẩn để đối soát:", [c for c in all_cols if c != pom_col])

        # 1. Tìm mẫu tương đương bằng AI
        vec_test = get_vector(target['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        matches = []
        for row in db.data:
            sim_img = cosine_similarity([vec_test], [row['vector']])[0][0] if vec_test and row['vector'] else 0
            matches.append({"data": row, "sim": sim_img})
        
        best_match = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
        ref_row = best_match['data']

        # 2. Xử lý so sánh đúng dòng
        st.subheader(f"📍 Kết quả so sánh với: {ref_row['file_name']} (Khớp {best_match['sim']*100:.1f}%)")
        
        # Ảnh đối chiếu
        im_col1, im_col2 = st.columns(2)
        with im_col1: st.image(target['img'], caption="File Kiểm", use_container_width=True)
        with im_col2: st.image(ref_row['image_url'], caption="File Mẫu (DB)", use_container_width=True)

        # Tạo bảng so sánh
        compare_list = []
        # Chuyển bảng mẫu từ DB thành DataFrame
        df_ref = pd.DataFrame(ref_row['spec_json'])
        
        for _, row in df_target.iterrows():
            pos_name = str(row[pom_col]).strip()
            val_new = parse_reitmans_val(row[size_col])
            
            # Tìm dòng tương ứng trong file mẫu bằng cách so khớp tên POM
            match_ref = df_ref[df_ref.iloc[:, 0].str.contains(re.escape(pos_name), case=False, na=False)]
            
            val_ref = 0
            if not match_ref.empty:
                # Ưu tiên lấy đúng cột size đã chọn, nếu không lấy cột giá trị đầu tiên
                ref_size_col = size_col if size_col in df_ref.columns else df_ref.columns[1]
                val_ref = parse_reitmans_val(match_ref.iloc[0][ref_size_col])

            diff = round(val_new - val_ref, 3)
            compare_list.append({
                "Description (Vị trí)": pos_name,
                "Actual (Kiểm)": val_new,
                "Standard (Mẫu)": val_ref,
                "Diff": diff,
                "Status": "✅ OK" if abs(diff) < 0.25 else "❌ LỆCH"
            })

        df_final = pd.DataFrame(compare_list)

        # 3. Hiển thị bảng màu sắc
        def color_diff(val):
            color = 'red' if val != 0 else 'black'
            return f'color: {color}'

        st.write("### 📊 BẢNG ĐỐI CHIẾU CHI TIẾT")
        st.dataframe(df_final.style.applymap(color_diff, subset=['Diff']), use_container_width=True, height=500)

        # Xuất Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, index=False, sheet_name='Audit_Report')
        st.download_button("📥 Tải báo cáo Excel", output.getvalue(), "audit_report.xlsx")

if st.button("LÀM MỚI"):
    st.rerun()
