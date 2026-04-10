import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Hãy điền đúng URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45.0", page_icon="🔍")

# --- LOAD MODEL AI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- UTILS ---
def parse_val(t):
    """Xử lý số phân số/hỗn số: 1 1/2, 3/4, 10.5..."""
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
    """Chuẩn hóa tên POM để soi đúng dòng"""
    if not t: return ""
    s = str(t).upper().strip()
    s = re.sub(r'[^A-Z0-9\s]', '', s)
    return " ".join(s.split())

# --- TRÍCH XUẤT (QUÉT ĐÚNG TRANG & ẢNH SIÊU NHẸ) ---
def extract_pom_pro(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=io.BytesIO(pdf_content))
        
        # 1. Ảnh JPG 50% siêu nhẹ (Matrix 0.8)
        if len(doc) > 0:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(0.8, 0.8))
            img_bytes = pix.tobytes("jpg", jpg_quality=50)

        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        # 2. Chỉ quét trang có từ khóa thông số
        POM_KEYS = ["POM", "POINT", "MEASURE", "DESCRIPTION", "DIMENSION"]
        VALUE_KEYS = ["NEW", "SPEC", "MEAS", "MEASURE", "SIZE", "SAMPLE", "TOL"]

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
                    for r_idx, row in df.head(8).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, c in enumerate(row_up):
                            if any(k in c for k in POM_KEYS): p_idx = i
                            if any(k in c for k in VALUE_KEYS): v_idx = i
                        if p_idx != -1 and v_idx != -1:
                            h_row = r_idx; break
                    
                    if h_row == -1: p_idx, v_idx = 0, 1

                    for i in range(h_row + 1, len(df)):
                        name = str(df.iloc[i][p_idx]).replace("\n", " ").strip().upper()
                        if len(name) < 3 or any(x in name for x in ["NOTE", "DATE"]): continue
                        val = parse_val(df.iloc[i][v_idx])
                        if val > 0: full_specs[name] = float(val)

        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.metric("Mẫu hiện có", res_count.count if res_count.count else 0)
    
    files = st.file_uploader("Nạp mẫu chuẩn", accept_multiple_files=True)
    if files and st.button("🚀 NẠP HỆ THỐNG"):
        for f in files:
            d = extract_pom_pro(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"thumbs/{f.name.replace('.pdf', '')}.jpg"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/jpeg", "x-upsert":"true"})
                
                res_url = supabase.storage.from_(BUCKET).get_public_url(path)
                img_url = res_url if isinstance(res_url, str) else getattr(res_url, "public_url", "")

                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'], 
                    "image_url": img_url, "category": d['brand']
                }).execute()
        st.success("Đã nạp xong!")
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V45.0")
t_file = st.file_uploader("Upload file PDF đối soát", type="pdf")

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
                    # 🔥 FIX LỖI TYPEERROR TẠI ĐÂY: Lấy phần tử [0][0] của ma trận similarity
                    sim_score = cosine_similarity(v_test, v_ref)[0][0]
                    matches.append({"data": i, "sim": float(sim_score) * 100})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Khớp mẫu: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ Kiểm", use_container_width=True)
                with c2: st.image(m['data']['image_url'], caption="Bản vẽ Mẫu", use_container_width=True)

                # --- SOI ĐÚNG DÒNG (ALIGNED COMPARISON) ---
                st.write("### 📊 Chi tiết đối soát dòng")
                ref_map = {normalize_pom(k): v for k, v in m['data']['spec_json'].items()}
                
                diff_data = []
                for p_name, v_target in target['specs'].items():
                    norm_name = normalize_pom(p_name)
                    v_ref = ref_map.get(norm_name, 0) # Soi đúng vị trí theo tên POM
                    diff = round(v_target - v_ref, 3)
                    
                    diff_data.append({
                        "Vị trí đo (POM)": p_name,
                        "Thực tế": v_target,
                        "Mẫu chuẩn": v_ref if v_ref > 0 else "N/A",
                        "Chênh lệch": diff,
                        "Kết quả": "✅ OK" if abs(diff) <= 0.125 else "❌ SAI"
                    })
                st.table(pd.DataFrame(diff_data))

if st.button("♻️ RESET"): st.rerun()
