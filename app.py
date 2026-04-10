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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V43.9", page_icon="📊")

# ================= MODEL AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# ================= UTILS =================
def parse_val(t):
    """Xử lý số Reitmans/Phân số: 1 1/2, 3/4..."""
    try:
        if t is None or str(t).lower() in ['nan', '', 'none', '-']: return 0
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def normalize_pom(t):
    """Chuẩn hóa tên POM để so khớp giữa các Brand"""
    if not t: return ""
    s = str(t).upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    return " ".join(s.split())

# ================= EXTRACT & CLEAN =================
def extract_tp_data(pdf_file):
    img_bytes, all_tables = None, []
    pdf_content = pdf_file.read()
    
    # 1. Image
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Tables
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tbs = page.extract_tables()
            for tb in tbs:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                # Fix Header & Clean NaN
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True)
                # QUAN TRỌNG: Thay thế NaN bằng chuỗi rỗng để không lỗi Supabase
                df = df.replace({np.nan: None}).fillna("") 
                all_tables.append(df)
                
    return {"img": img_bytes, "tables": all_tables}

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            return model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU MẪU")
    up_files = st.file_uploader("Nạp Techpack chuẩn", accept_multiple_files=True)
    if up_files and st.button("🚀 LƯU VÀO HỆ THỐNG"):
        for f in up_files:
            d = extract_tp_data(f)
            if d['tables']:
                main_df = max(d['tables'], key=len)
                vec = get_vector(d['img'])
                
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                if not isinstance(img_url, str): img_url = getattr(img_url, "public_url", "")

                # Insert an toàn với spec_json đã lọc NaN
                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": main_df.to_dict('records'),
                    "image_url": img_url
                }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V43.9")
f_target = st.file_uploader("📤 Upload file đối soát", type="pdf")

if f_target:
    target_data = extract_tp_data(f_target)
    if target_data['tables']:
        df_target = target_data['tables'][0] # Lấy bảng đầu
        cols = df_target.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1: pom_col = st.selectbox("Cột Vị trí (POM):", cols, index=0)
        with c2: size_col = st.selectbox("Chọn Size Audit:", [c for c in cols if c != pom_col])

        # Tìm mẫu AI
        v_test = get_vector(target_data['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        if db.data:
            matches = []
            for row in db.data:
                sim = cosine_similarity([v_test], [row['vector']])[0][0] * 100 if row['vector'] else 0
                matches.append({"row": row, "sim": sim})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            ref_row = best['row']
            df_ref = pd.DataFrame(ref_row['spec_json'])

            st.divider()
            st.subheader(f"✨ Khớp mẫu: {ref_row['file_name']} ({best['sim']:.1f}%)")

            # Ảnh Side-by-side
            im1, im2 = st.columns(2)
            with im1: st.image(target_data['img'], caption="File Kiểm")
            with im2: st.image(ref_row['image_url'], caption="File Mẫu")

            # --- LOGIC SOI ĐÚNG DÒNG ---
            ref_map = {}
            ref_cols = df_ref.columns
            for _, r in df_ref.iterrows():
                # Map theo tên POM đã chuẩn hóa
                key = normalize_pom(r[ref_cols[0]])
                ref_map[key] = parse_val(r.get(size_col, r.iloc[1] if len(r)>1 else 0))

            results = []
            for _, rt in df_target.iterrows():
                p_name = rt[pom_col]
                norm_p = normalize_pom(p_name)
                if not norm_p: continue
                
                v_new = parse_val(rt[size_col])
                v_ref = ref_map.get(norm_p, 0)
                diff = round(v_new - v_ref, 3)

                results.append({
                    "Description": p_name,
                    "Kiểm (New)": v_new,
                    "Mẫu (Ref)": v_ref,
                    "Lệch": diff,
                    "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })

            st.table(pd.DataFrame(results)) # Hiển thị đúng dòng đối xứng
