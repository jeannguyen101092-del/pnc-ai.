import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45.9", page_icon="🔍")

# --- MODEL AI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- UTILS ---
def parse_val(t):
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
    """Chuẩn hóa POM để dù Brand nào cũng soi đúng dòng"""
    if not t: return ""
    s = str(t).upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s) # Chỉ giữ chữ và số để khớp tuyệt đối
    return s

def extract_pom_pro(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=io.BytesIO(pdf_content))
        if len(doc) > 0:
            # Ảnh JPG siêu nhẹ (Matrix 0.6)
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(0.6, 0.6))
            img_bytes = pix.tobytes("jpg", jpg_quality=30)
        
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        POM_KEYS = ["POM", "POINT", "MEASURE", "DESCRIPTION", "DIMENSION", "SPEC"]
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text_page = (page.extract_text() or "").upper()
                if not any(k in text_page for k in POM_KEYS): continue

                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    h_row, p_idx, v_idx = -1, -1, -1
                    for r_idx, row in df.head(8).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, c in enumerate(row_up):
                            if any(k in c for k in POM_KEYS): p_idx = i
                            if any(k in c for k in ["NEW", "SPEC", "SIZE", "SAMPLE"]): v_idx = i
                        if p_idx != -1 and v_idx != -1:
                            h_row = r_idx; break
                    
                    if h_row == -1: p_idx, v_idx = 0, 1

                    for i in range(h_row + 1, len(df)):
                        name = str(df.iloc[i][p_idx]).replace("\n", " ").strip().upper()
                        if len(name) < 3: continue
                        val = parse_val(df.iloc[i][v_idx])
                        if val > 0: full_specs[name] = float(val)
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR: KHO DỮ LIỆU ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    try:
        res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Mẫu hiện có", res.count if res.count else 0)
    except: st.error("⚠️ Lỗi kết nối SQL")
    
    files = st.file_uploader("Nạp mẫu chuẩn", accept_multiple_files=True)
    if files and st.button("🚀 NẠP HỆ THỐNG"):
        for f in files:
            d = extract_pom_pro(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().astype(float).tolist()
                
                path = f"thumbs/{f.name.replace('.pdf', '')}.jpg"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/jpeg", "x-upsert":"true"})
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'], 
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V45.9 PRO")
t_file = st.file_uploader("Upload file PDF đối soát", type="pdf")

if t_file:
    target = extract_pom_pro(t_file)
    if target and target['specs']:
        st.info(f"Đã trích xuất {len(target['specs'])} thông số. Đang tìm mẫu khớp...")
        
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().astype(float)
            
            matches = []
            for i in db_res.data:
                v_ref = i.get('vector')
                if v_ref and len(v_ref) == 512:
                    v1, v2 = np.array(v_test), np.array(v_ref)
                    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
                    sim = np.dot(v1, v2) / denom if denom != 0 else 0
                    matches.append({"data": i, "sim": float(sim) * 100})
            
            if matches:
                # Lấy mẫu có độ tương đồng cao nhất
                m = sorted(matches, key=lambda x: x['sim'], reverse=True)
                res_best = m[0]
                
                st.success(f"🏆 Khớp mẫu: **{res_best['data']['file_name']}** ({res_best['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ Kiểm", use_container_width=True)
                with c2: st.image(res_best['data']['image_url'], caption="Mẫu gốc", use_container_width=True)

                # --- SOI ĐÚNG DÒNG (ALIGNED) ---
                st.write("### 📊 Chi tiết đối soát thông số")
                ref_map = {normalize_pom(k): v for k, v in res_best['data']['spec_json'].items()}
                
                diff_list = []
                for p_name, v_target in target['specs'].items():
                    p_key = normalize_pom(p_name)
                    v_ref = ref_map.get(p_key, 0)
                    diff = round(v_target - v_ref, 3)
                    
                    diff_list.append({
                        "Vị trí đo (POM)": p_name,
                        "Thực tế": v_target,
                        "Mẫu chuẩn": v_ref if v_ref > 0 else "N/A",
                        "Chênh lệch": diff,
                        "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                    })
                st.table(pd.DataFrame(diff_list))
            else:
                st.error("Không tìm thấy mẫu tương đồng. Hãy nạp lại kho dữ liệu!")
