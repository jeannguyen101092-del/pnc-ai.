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
        # Bắt các định dạng: "1 1/2", "3/4", "1.5", "10"
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v: # Hỗn số "1 1/2"
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT MULTI-SIZE =================
def extract_pom_pro(pdf_file):
    brand = "OTHER"
    img_bytes = None
    pdf_content = pdf_file.read()
    
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += (page.get_text() or "").upper() + " "
    if "REITMANS" in all_text: brand = "REITMANS"
    
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    table_data = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                # Xử lý Header
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True)
                table_data.append(df)

    return {"img": img_bytes, "tables": table_data, "brand": brand}

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

# ================= SIDEBAR: NẠP DỮ LIỆU =================
with st.sidebar:
    st.header("📂 HỆ THỐNG DỮ LIỆU")
    count_res = supabase.table("ai_data").select("*", count="exact").execute()
    st.metric("Tổng số mẫu", count_res.count if count_res.count else 0)
    
    upload_files = st.file_uploader("Nạp Techpack Mẫu", accept_multiple_files=True)
    if upload_files and st.button("🚀 LƯU VÀO DB"):
        for f in upload_files:
            data = extract_pom_pro(f)
            # Lưu bảng đầu tiên làm mẫu chuẩn
            specs = data['tables'][0].to_dict('records') if data['tables'] else []
            vec = get_vector(data['img'])
            
            img_url = ""
            if data['img']:
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                # FIX LỖI TYPEERROR TẠI ĐÂY
                res_url = supabase.storage.from_(BUCKET).get_public_url(path)
                img_url = res_url if isinstance(res_url, str) else getattr(res_url, "public_url", "")

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": specs,
                "image_url": img_url
            }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# ================= MAIN: AUDIT =================
st.title("🔥 AI AUDITOR V49 - SMART COMPARISON")

file_test = st.file_uploader("📤 Upload file PDF kiểm tra", type="pdf")

if file_test:
    target = extract_pom_pro(file_test)
    
    if not target['tables']:
        st.error("Không tìm thấy bảng thông số.")
    else:
        df_target = target['tables'][0] 
        all_cols = df_target.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"Brand: **{target['brand']}**")
            pom_col = st.selectbox("Cột Vị trí (POM):", all_cols, index=0)
        with c2:
            size_col = st.selectbox("Chọn Size đối soát:", [c for c in all_cols if c != pom_col])

        # Tìm mẫu AI
        vec_test = get_vector(target['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        results = []
        for row in db.data:
            sim_img = cosine_similarity([vec_test], [row['vector']])[0][0] if vec_test and row['vector'] else 0
            results.append({"data": row, "sim": sim_img})
        
        if results:
            best = sorted(results, key=lambda x: x['sim'], reverse=True)[0]
            ref_row = best['data']

            st.divider()
            st.subheader(f"📍 Đối chiếu với: {ref_row['file_name']} (Khớp hình ảnh: {best['sim']*100:.1f}%)")
            
            im1, im2 = st.columns(2)
            with im1: st.image(target['img'], caption="Ảnh Kiểm", use_container_width=True)
            with im2: st.image(ref_row['image_url'], caption="Ảnh Mẫu DB", use_container_width=True)

            # So sánh đúng dòng
            df_ref = pd.DataFrame(ref_row['spec_json'])
            compare_data = []
            
            for _, r_target in df_target.iterrows():
                p_name = str(r_target[pom_col]).strip()
                v_new = parse_reitmans_val(r_target[size_col])
                
                # Tìm dòng tương ứng ở file mẫu (Search Description)
                match_ref = df_ref[df_ref.iloc[:, 0].astype(str).str.contains(re.escape(p_name), case=False, na=False)]
                
                v_ref = 0
                if not match_ref.empty:
                    # Lấy cột cùng tên size hoặc lấy cột giá trị đầu tiên sau cột Description
                    target_col_ref = size_col if size_col in df_ref.columns else df_ref.columns[1]
                    v_ref = parse_reitmans_val(match_ref.iloc[0][target_col_ref])

                diff = round(v_new - v_ref, 3)
                compare_data.append({
                    "Vị trí đo (POM)": p_name,
                    "File Kiểm": v_new,
                    "Mẫu chuẩn": v_ref,
                    "Chênh lệch": diff,
                    "Kết quả": "✅ OK" if abs(diff) < 0.1 else "❌ LỆCH"
                })

            df_final = pd.DataFrame(compare_data)
            
            st.write("### 📊 BẢNG CHI TIẾT THEO DÒNG")
            def color_diff(val):
                return 'background-color: #ffcccc' if val != 0 else ''

            st.dataframe(df_final.style.applymap(color_diff, subset=['Chênh lệch']), use_container_width=True, height=600)

            # Export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_final.to_excel(writer, index=False)
            st.download_button("📥 Tải báo cáo Excel", output.getvalue(), f"Audit_{ref_row['file_name']}.xlsx")

if st.button("RESET"):
    st.rerun()
