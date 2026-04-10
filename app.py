import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V44.2", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- UTILS ---
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    """Chuẩn hóa tên để so khớp đúng dòng"""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT (QUÉT ĐÚNG TRANG & TỐI ƯU ẢNH) ---
def extract_pom_pro(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")

        # 1. Tối ưu ảnh: JPG 60% chất lượng cho siêu nhẹ
        if len(doc) > 0:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            img_bytes = pix.tobytes("jpg", jpg_quality=60)

        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        # 2. Quét bảng (Chỉ quét trang có từ khóa thông số)
        POM_KEYS = ["POM", "POINT", "MEASURE", "DESCRIPTION", "DIMENSION"]
        VALUE_KEYS = ["NEW", "SPEC", "MEAS", "MEASURE", "GARMENT", "SIZE", "SAMPLE"]

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_page = (page.extract_text() or "").upper()
                if not any(k in text_page for k in POM_KEYS): continue

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue

                    # Tìm Header tự động
                    h_row, p_idx, v_idx = -1, -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, c in enumerate(row_up):
                            if any(k in c for k in POM_KEYS): p_idx = i
                            if any(k in c for k in VALUE_KEYS): v_idx = i
                        if p_idx != -1 and v_idx != -1:
                            h_row = r_idx
                            break
                    
                    if h_row == -1: p_idx, v_idx = 0, 1

                    for i in range(h_row + 1, len(df)):
                        name = str(df.iloc[i][p_idx]).replace("\n", " ").strip().upper()
                        if len(name) < 3 or any(x in name for x in ["NOTE", "TOL"]): continue
                        val = parse_reitmans_val(df.iloc[i][v_idx])
                        if val > 0: full_specs[name] = val

        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR (FIX LỖI APIError) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    files = st.file_uploader("Nạp mẫu chuẩn", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            d = extract_pom_pro(f)
            if d and d['specs']:
                # AI Vector
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                # Lưu ảnh nhẹ JPG
                path = f"thumbs/{f.name.replace('.pdf', '.jpg')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/jpeg", "x-upsert":"true"})
                
                # FIX LỖI NaN & Insert
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, 
                    "spec_json": d['specs'], # Ở đây d['specs'] là Dict sạch
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "category": d['brand']
                }).execute()
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN (SOI ĐÚNG VỊ TRÍ) ---
st.title("🔍 AI Fashion Auditor V44.2")
t_file = st.file_uploader("Upload file kiểm tra", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_pro(t_file)
    if target and target['specs']:
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": i, "sim": sim})
            
            m = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
            st.subheader(f"✨ Khớp mẫu: {m['data']['file_name']} ({m['sim']:.1f}%)")
            
            c1, c2 = st.columns(2)
            with c1: st.image(target['img'], caption="File Kiểm")
            with c2: st.image(m['data']['image_url'], caption="File Mẫu")

            # --- SOI ĐÚNG DÒNG (ALIGNED COMPARISON) ---
            ref_specs = m['data']['spec_json']
            ref_map = {clean_pos(k): v for k, v in ref_specs.items()}
            
            diff_list = []
            for p_name, v_target in target['specs'].items():
                p_key = clean_pos(p_name)
                v_ref = ref_map.get(p_key, 0)
                diff = round(v_target - v_ref, 3)
                
                diff_list.append({
                    "Vị trí đo (POM)": p_name,
                    "Thực tế (New)": v_target,
                    "Mẫu (Ref)": v_ref,
                    "Lệch": diff,
                    "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                })
            st.table(pd.DataFrame(diff_list))

if st.button("RESET"): st.rerun()
