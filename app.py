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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V7 PRO", page_icon="👖")

# ================= UTILS =================
def parse_reitmans_val(t):
    try:
        if t is None or str(t).lower() in ['nan', '-', 'none', '']: return 0
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

model_ai = load_ai()

def get_vector(img_bytes):
    if not img_bytes: return [0.0]*512
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        with torch.no_grad():
            v = model_ai(tf(img).unsqueeze(0)).view(-1).numpy()
            v = v / (np.linalg.norm(v) + 1e-7)
        return v.tolist()
    except: return [0.0]*512

def extract_data(pdf_file):
    img_bytes = None
    pdf_content = pdf_file.read()
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    tables = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tbs = page.extract_tables()
            for tb in tbs:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                df.columns = [str(c).replace('\n',' ').strip().upper() for c in df.iloc[0]]
                df = df[1:].reset_index(drop=True).fillna("")
                tables.append(df)
    return {"img": img_bytes, "tables": tables}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 QUẢN TRỊ KHO")
    files = st.file_uploader("NẠP KHO (PDF)", accept_multiple_files=True)
    if files and st.button("🚀 NẠP"):
        for f in files:
            d = extract_data(f)
            vec = get_vector(d['img'])
            # Lấy bảng có nhiều dữ liệu nhất để làm mẫu
            main_df = max(d['tables'], key=len) if d['tables'] else pd.DataFrame()
            specs = main_df.to_dict('records')
            
            img_url = ""
            if d['img']:
                path = f"lib/{f.name}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                res_url = supabase.storage.from_(BUCKET).get_public_url(path)
                img_url = res_url if isinstance(res_url, str) else getattr(res_url, "public_url", "")
            
            supabase.table("ai_data").insert({
                "file_name": f.name, "vector": vec, "spec_json": specs, "image_url": img_url
            }).execute()
        st.success("Đã nạp thành công!")
        st.rerun()

# ================= MAIN =================
st.title("👖 AI Fashion Pro V7 PRO")

f_test = st.file_uploader("Upload PDF kiểm tra", type="pdf")

if f_test:
    target = extract_data(f_test)
    if target['tables']:
        df_target = target['tables'][0] # Lấy bảng đầu tiên
        cols = df_target.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1: pom_col = st.selectbox("Cột Vị trí (POM):", cols, index=0)
        with c2: size_col = st.selectbox("Chọn Size kiểm:", [c for c in cols if c != pom_col])

        # Tìm mẫu AI
        v_test = get_vector(target['img'])
        db = supabase.table("ai_data").select("*").execute()
        
        if db.data:
            matches = []
            for row in db.data:
                sim = cosine_similarity([v_test], [row['vector']])[0][0] if row['vector'] else 0
                matches.append({"row": row, "sim": sim})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            ref_row = best['row']
            
            # --- FIX VALUEERROR TẠI ĐÂY ---
            spec_data = ref_row['spec_json']
            df_ref = pd.DataFrame(spec_data) if isinstance(spec_data, list) else pd.DataFrame([spec_data])

            st.info(f"🏆 Mẫu khớp nhất: **{ref_row['file_name']}** ({best['sim']*100:.1f}%)")

            # Ảnh 2 bên
            im_c1, im_c2 = st.columns(2)
            with im_c1: st.image(target['img'], caption="ẢNH KIỂM", use_container_width=True)
            with im_c2: st.image(ref_row['image_url'], caption="ẢNH MẪU", use_container_width=True)

            # --- SO SÁNH ĐÚNG DÒNG (MAP THEO POM) ---
            st.write("### 📊 BẢNG ĐỐI CHIẾU THÔNG SỐ")
            
            # Tạo bản đồ giá trị từ file mẫu
            ref_map = {}
            if not df_ref.empty:
                # Giả định cột đầu tiên của file mẫu là tên POM
                ref_pom_key = df_ref.columns[0]
                for _, r in df_ref.iterrows():
                    p_key = str(r[ref_pom_key]).strip().upper()
                    ref_map[p_key] = parse_reitmans_val(r.get(size_col, r.iloc[1] if len(r)>1 else 0))

            compare_results = []
            for _, r_t in df_target.iterrows():
                pom_name = str(r_t[pom_col]).strip()
                if not pom_name: continue
                
                val_new = parse_reitmans_val(r_t[size_col])
                val_ref = ref_map.get(pom_name.upper(), 0)
                diff = round(val_new - val_ref, 3)

                compare_results.append({
                    "VỊ TRÍ ĐO (POM)": pom_name,
                    "KIỂM (NEW)": val_new,
                    "MẪU (REF)": val_ref,
                    "LỆCH": diff,
                    "KẾT QUẢ": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })

            res_df = pd.DataFrame(compare_results)
            st.table(res_df) # Hiển thị bảng đối xứng đúng dòng

if st.button("LÀM MỚI"): st.rerun()
