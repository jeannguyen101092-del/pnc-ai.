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
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= EXTRACT =================
def extract_pom_pro(pdf_file):
    img_bytes = None
    pdf_content = pdf_file.read()
    
    # 1. Lấy ảnh
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Đọc bảng
    table_data = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True)
                # Xử lý NaN để tránh lỗi Supabase
                df = df.fillna("") 
                table_data.append(df)
    return {"img": img_bytes, "tables": table_data}

def get_vector(img_bytes):
    if not img_bytes: return [0.0] * 512
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        with torch.no_grad():
            v = model_ai(tf(img).unsqueeze(0)).view(-1).numpy()
            v = v / (np.linalg.norm(v) + 1e-7)
        return v.tolist()
    except: return [0.0] * 512

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 HỆ THỐNG DỮ LIỆU")
    files = st.file_uploader("Nạp Techpack Mẫu", accept_multiple_files=True)
    if files and st.button("🚀 LƯU VÀO DB"):
        for f in files:
            d = extract_pom_pro(f)
            vec = get_vector(d['img'])
            # Chỉ lấy bảng đầu tiên làm mẫu, xóa bỏ NaN
            specs = d['tables'][0].to_dict('records') if d['tables'] else []
            
            img_url = ""
            if d['img']:
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                res_url = supabase.storage.from_(BUCKET).get_public_url(path)
                img_url = res_url if isinstance(res_url, str) else getattr(res_url, "public_url", "")

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": specs,
                "image_url": img_url
            }).execute()
        st.success("✅ Thành công!")
        st.rerun()

# ================= MAIN =================
st.title("🔥 AI AUDITOR V49")

file_test = st.file_uploader("📤 Upload file kiểm tra", type="pdf")

if file_test:
    target = extract_pom_pro(file_test)
    if target['tables']:
        df_target = target['tables'][0]
        cols = df_target.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1: pom_col = st.selectbox("Cột Vị trí (POM):", cols, index=0)
        with c2: size_col = st.selectbox("Chọn Size đối soát:", [c for c in cols if c != pom_col])

        # Tìm mẫu AI
        v_test = get_vector(target['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        matches = []
        for row in db.data:
            sim = cosine_similarity([v_test], [row['vector']])[0][0] if row['vector'] else 0
            matches.append({"data": row, "sim": sim})
        
        if matches:
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            ref_row = best['data']
            df_ref = pd.DataFrame(ref_row['spec_json'])

            st.subheader(f"📍 Đối chiếu: {ref_row['file_name']} ({best['sim']*100:.1f}%)")
            
            # --- SO SÁNH ĐÚNG DÒNG ---
            compare_list = []
            for _, r in df_target.iterrows():
                p_name = str(r[pom_col]).strip()
                val_new = parse_reitmans_val(r[size_col])
                
                # Tìm dòng tương ứng ở mẫu chuẩn (Dựa trên tên POM)
                match_ref = df_ref[df_ref.iloc[:, 0].astype(str).str.contains(re.escape(p_name), case=False, na=False)]
                
                val_ref = 0
                if not match_ref.empty:
                    # Lấy cột cùng tên size hoặc cột số 1
                    ref_col = size_col if size_col in df_ref.columns else df_ref.columns[1]
                    val_ref = parse_reitmans_val(match_ref.iloc[0][ref_col])

                diff = round(val_new - val_ref, 3)
                compare_list.append({
                    "Vị trí (POM)": p_name,
                    "Thực tế": val_new,
                    "Mẫu chuẩn": val_ref,
                    "Lệch": diff,
                    "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })

            st.table(pd.DataFrame(compare_list)) # Hiển thị bảng cố định đúng dòng

            col_im1, col_im2 = st.columns(2)
            with col_im1: st.image(target['img'], caption="Ảnh Kiểm")
            with col_im2: st.image(ref_row['image_url'], caption="Ảnh Mẫu")

st.button("LÀM MỚI", on_click=lambda: st.rerun())
