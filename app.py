import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V45.0", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except: return None
model_ai = load_ai()

def parse_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ['nan', '-', 'none', 'tbd', 'tol']): return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT ĐÚNG TRANG POM & HÌNH ẢNH ---
def extract_pom_v450(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Tìm trang có hình ảnh hoặc trang đầu tiên làm thumbnail
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").upper()
                # BỎ QUA các trang chứa từ khóa Nguyên phụ liệu (BOM)
                if any(x in text for x in ["POLY CORE", "SEWING THREAD", "BUTTON", "ZIPPER", "HANGTAG"]):
                    continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    desc_col, val_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        # Tìm cột Description
                        for i, cell in enumerate(row_up):
                            if any(k in cell for k in ["DESCRIPTION", "POM NAME", "POINT OF MEASURE"]):
                                desc_col = i
                                break
                        # Tìm cột Size mẫu (Ưu tiên NEW của Reitmans hoặc M, L, SAMPLE)
                        for i, cell in enumerate(row_up):
                            if i == desc_col: continue
                            if any(k in cell for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " M ", " L "]):
                                val_col = i
                                break
                        
                        if desc_col != -1 and val_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[desc_col]).replace('\n',' ').strip().upper()
                                # Lọc dòng rác và BOM rò rỉ
                                if len(name) < 4 or any(x in name for x in ["CORE", "POLY", "TAPE", "THREAD"]): continue
                                
                                val_num = parse_val(d_row[val_col])
                                if val_num > 0: full_specs[name] = val_num
                            break
        doc.close()
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            d = extract_pom_v450(f)
            if d and d['specs']:
                # Upload ảnh lên Storage
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                # Tính Vector
                vec = None
                if model_ai:
                    img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'], "image_url": img_url
                }).execute()
                st.success(f"Đã nạp đúng bảng POM: {f.name}")
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V45.0")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v450(t_file)
    if target and target['specs']:
        st.success(f"✅ Đã quét trang POM: Tìm thấy {len(target['specs'])} thông số đo.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            # Lấy mẫu khớp nhất (tạm thời lấy mẫu cuối nếu chưa có logic vector hoàn chỉnh)
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu: {m['file_name']}")
            
            c1, c2 = st.columns(2)
            with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
            with c2: st.image(m['image_url'], caption="Mẫu gốc trong kho")

            diff_list = []
            for p_name, v_target in target['specs'].items():
                v_ref = 0
                p_clean = clean_pos(p_name)
                for k_ref, val_ref in m['spec_json'].items():
                    if p_clean == clean_pos(k_ref):
                        v_ref = val_ref
                        break
                diff_list.append({"Hạng mục": p_name, "Kiểm tra": v_target, "Mẫu gốc": v_ref, "Lệch": round(v_target - v_ref, 2) if v_ref > 0 else "N/A"})
            
            if diff_list:
                st.table(pd.DataFrame(diff_list).style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
