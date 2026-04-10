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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V44.0", page_icon="📊")

# ================= MODEL AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# ================= UTILS =================
def parse_reitmans_val(t):
    """Xử lý số Reitmans/Phân số: 1 1/2, 3/4, 10.5..."""
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
    """Chuẩn hóa tên POM để soi đúng dòng giữa các Brand"""
    if not t: return ""
    s = str(t).upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s) # Bỏ ký tự đặc biệt
    return " ".join(s.split())

# ================= EXTRACT & CLEAN =================
def extract_tp_data(pdf_file):
    img_bytes, all_tables = None, []
    pdf_content = pdf_file.read()
    
    # 1. Lấy ảnh
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Lấy bảng và xử lý sạch dữ liệu
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tbs = page.extract_tables()
            for tb in tbs:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                
                # Header & Clean
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True)
                
                # FIX LỖI APIError: Chuyển NaN thành None/Chuỗi rỗng
                df = df.replace({np.nan: None}) 
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
    
    if up_files and st.button("🚀 NẠP HỆ THỐNG"):
        for f in up_files:
            d = extract_tp_data(f)
            if d['tables']:
                main_df = max(d['tables'], key=len)
                vec = get_vector(d['img'])
                
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                if not isinstance(img_url, str): img_url = getattr(img_url, "public_url", "")

                # Nạp dữ liệu đã xử lý NaN
                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": main_df.to_dict('records'),
                    "image_url": img_url
                }).execute()
        st.success("✅ Đã nạp thành công!")
        st.rerun()

# ================= MAIN AUDIT =================
st.title("🔍 AI Fashion Auditor V44.0")
f_target = st.file_uploader("📤 Upload file đối soát", type="pdf")

if f_target:
    target_data = extract_tp_data(f_target)
    if target_data['tables']:
        # Lấy bảng thông số đầu tiên
        df_target = target_data['tables'][0]
        cols = df_target.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1: pom_col = st.selectbox("Cột Tên Vị Trí (POM):", cols, index=0)
        with c2: size_col = st.selectbox("Cột Size cần Audit:", [c for c in cols if c != pom_col])

        # Tìm mẫu AI tương đồng
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

            # Ảnh so sánh 2 bên
            im1, im2 = st.columns(2)
            with im1: st.image(target_data['img'], caption="Bản vẽ Kiểm tra")
            with im2: st.image(ref_row['image_url'], caption="Bản vẽ Mẫu Gốc")

            # --- LOGIC SOI ĐÚNG DÒNG (ALIGNED COMPARISON) ---
            st.write(f"### 📊 Chi tiết đối soát dòng (Size: {size_col})")
            
            # Map dữ liệu từ file mẫu (Normalize POM -> Value)
            ref_map = {}
            # Lấy cột đầu tiên của file mẫu làm tên POM chuẩn
            ref_pom_key = df_ref.columns[0] 
            for _, r in df_ref.iterrows():
                k = normalize_pom(r[ref_pom_key])
                # Ưu tiên cột cùng tên size, nếu không bốc cột giá trị đầu tiên sau POM
                val = parse_reitmans_val(r.get(size_col, r.iloc[1] if len(r)>1 else 0))
                ref_map[k] = val

            audit_results = []
            for _, rt in df_target.iterrows():
                p_raw = rt[pom_col]
                norm_p = normalize_pom(p_raw)
                if not norm_p: continue
                
                val_new = parse_reitmans_val(rt[size_col])
                val_ref = ref_map.get(norm_p, 0)
                diff = round(val_new - val_ref, 3)

                audit_results.append({
                    "POM / Description": p_raw,
                    "KIỂM (Actual)": val_new,
                    "MẪU (Standard)": val_ref,
                    "LỆCH (Diff)": diff,
                    "KẾT QUẢ": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })

            # Hiển thị bảng đối soát
            res_df = pd.DataFrame(audit_results)
            st.table(res_df) 

st.divider()
if st.button("♻️ Làm mới hệ thống"): st.rerun()
