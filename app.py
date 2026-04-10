import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Kiểm tra kỹ URL và KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V45.4", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    except: return None
model_ai = load_ai()

# --- HÀM PARSE SỐ (GIỮ CHUẨN REITMANS) ---
def parse_val(t):
    try:
        if t is None: return 0
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ['nan', '-', 'none', 'tol']): return 0
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

# --- TRÍCH XUẤT THÔNG SỐ (VÉT SẠCH TRANG POM/SPEC) ---
def extract_pom_v454(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "").upper()
                if "POLY CORE" in text or "SEWING THREAD" in text: continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    p_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() if c else "" for c in row]
                        for i, c in enumerate(row_up):
                            if any(k in c for k in ["POM NAME", "DESCRIPTION", "ITEM"]): p_col = i
                        for i, c in enumerate(row_up):
                            if i != p_col and any(k in c for k in ["NEW", "FINAL", "SAMPLE", "SPEC", " M "]): v_col = i
                        
                        if p_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_col]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["DATE", "PAGE"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: full_specs[name] = val
                            break
        doc.close()
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR: NẠP FILE (CƠ CHẾ ÉP NẠP) ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    try:
        res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Mẫu đã nạp", res_c.count if res_c.count is not None else 0)
    except: st.error("Lỗi kết nối database")

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        for f in files:
            d = extract_pom_v454(f)
            if d and d['specs']:
                img_url, vec = "", None
                try:
                    path = f"lib_{f.name.replace('.pdf', '.png')}"
                    supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"x-upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                    if model_ai:
                        img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                except: pass

                # NẠP THỬ LẦN 1: Đầy đủ
                try:
                    supabase.table("ai_data").insert({"file_name":f.name, "spec_json":d['specs'], "vector":vec, "image_url":img_url, "category":"GENERIC"}).execute()
                    st.success(f"Nạp xong: {f.name}")
                except Exception as e1:
                    # NẠP THỬ LẦN 2: Bỏ cột category và vector (Nếu DB cũ chưa cập nhật)
                    try:
                        supabase.table("ai_data").insert({"file_name":f.name, "spec_json":d['specs'], "image_url":img_url}).execute()
                        st.warning(f"Đã nạp {f.name} (Chế độ tối giản)")
                    except Exception as e2:
                        st.error(f"❌ THẤT BẠI: {f.name}")
                        st.code(f"Lỗi 1: {e1}\nLỗi 2: {e2}")
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V44.4")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v454(t_file)
    if target and target['specs']:
        st.success(f"✅ Tìm thấy {len(target['specs'])} thông số.")
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            m = db_res.data[-1] 
            st.subheader(f"✨ Đối chiếu: {m['file_name']}")
            c1, c2 = st.columns(2)
            if target['img']: with c1: st.image(target['img'], caption="Kiểm tra")
            if m['image_url']: with c2: st.image(m['image_url'], caption="Gốc")

            df_r = pd.DataFrame([{"Hạng mục": k, "Kiểm tra": v, "Mẫu gốc": m['spec_json'].get(k, 0), "Lệch": round(v - m['spec_json'].get(k, 0), 2) if m['spec_json'].get(k, 0) > 0 else "N/A"} for k, v in target['specs'].items()])
            st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if isinstance(x, (int, float)) and abs(x) > 0.5 else 'color: white', subset=['Lệch']))
            
            out = io.BytesIO()
            df_r.to_excel(out, index=False)
            st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Report_{m['file_name']}.xlsx")
