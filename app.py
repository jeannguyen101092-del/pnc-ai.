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

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("Lỗi kết nối Supabase!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V5.1", page_icon="📏")

@st.cache_resource
def load_ai_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai_model()

def parse_measurement(text):
    try:
        txt = str(text).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null', '0']: return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        val_str = match[0]
        if ' ' in val_str:
            p = val_str.split()
            return float(p[0]) + eval(p[1])
        return eval(val_str) if '/' in val_str else float(val_str)
    except: return 0

def clean_text(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

def extract_specs_from_pdf(pdf_file):
    full_specs, img_bytes = {}, None
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    desc_col, val_col = -1, -1
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if any(x in row_up for x in ["DESCRIPTION", "POM NAME"]):
                            desc_col = next((i for i, v in enumerate(row_up) if "DESCRIPTION" in v or "POM NAME" in v), -1)
                            # Ưu tiên tìm cột Size 32 hoặc NEW
                            for i, v in enumerate(row_up):
                                if v in ["32", "NEW", "SAMPLE", "SPEC"]: val_col = i; break
                            
                            if desc_col != -1 and val_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[desc_col]).strip().upper()
                                    if len(name) < 3 or "TOL" in name: continue
                                    val = parse_measurement(d_row[val_col])
                                    if val > 0: full_specs[name] = val
                                break
        return {"specs": full_specs, "img": img_bytes}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 THƯ VIỆN MẪU")
    files = st.file_uploader("Nạp Techpack Gốc", accept_multiple_files=True)
    if files and st.button("🚀 LƯU VÀO KHO"):
        p = st.progress(0)
        for i, f in enumerate(files):
            d = extract_specs_from_pdf(f)
            if d and d['specs'] and d['img']:
                img = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
                
                # FIX LỖI TÊN FILE: replace dấu chấm để tránh lỗi path
                f_safe_name = f.name.replace(".", "_")
                f_path = f"lib_{f_safe_name}.png"
                
                supabase.storage.from_(BUCKET).upload(path=f_path, file=d['img'], file_options={"x-upsert":"true", "content-type":"image/png"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(f_path)
                }).execute()
            p.progress((i + 1) / len(files))
        st.success("Đã nạp xong!")

# --- PHẦN CHÍNH ---
t_file = st.file_uploader("Tải file CẦN KIỂM TRA", type="pdf")
if t_file:
    target = extract_specs_from_pdf(t_file)
    if target and target['specs']:
        st.success(f"Tìm thấy {len(target['specs'])} thông số.")
        db = supabase.table("ai_data").select("*").execute()
        if db.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                # CHUẨN HÓA 2D: .reshape(1, -1)
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for item in db.data:
                if item.get('vector'):
                    # FIX LỖI VALUEERROR: Đảm bảo v_ref cũng là 2D
                    v_ref = np.array(item['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": item, "sim": sim})
            
            if matches:
                best = sorted(matches, key=lambda x: x['sim'], reverse=True)
                m = best[0]
                st.subheader(f"✨ Khớp: {m['data']['file_name']} ({m['sim']:.1f}%)")
                
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc")

                diff_data = []
                ref_specs = m['data']['spec_json']
                ref_map = {clean_text(k): v for k, v in ref_specs.items()}

                for p_name, v_target in target['specs'].items():
                    v_ref = ref_map.get(clean_text(p_name), 0)
                    diff = round(v_target - v_ref, 3)
                    diff_data.append({
                        "Hạng mục": p_name, "Kiểm tra": v_target, "Gốc": v_ref, 
                        "Lệch": diff, "Kết quả": "🚩 Lệch" if abs(diff) > 0.01 else "✔️ OK"
                    })
                
                st.table(pd.DataFrame(diff_data))
