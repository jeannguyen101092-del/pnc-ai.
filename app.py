import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- 1. CẤU HÌNH (Thay thông tin của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("Chưa kết nối được Supabase.")

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V5.2", page_icon="📏")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null', '0']: return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_txt(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

def extract_pdf(pdf_file):
    specs, img_b = {}, None
    try:
        content = pdf_file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        if len(doc) > 0:
            img_b = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    d_col, v_col = -1, -1
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if any(x in row_up for x in ["DESCRIPTION", "POM NAME"]):
                            d_col = next((i for i, v in enumerate(row_up) if "DESCRIPTION" in v or "POM NAME" in v), -1)
                            for i, v in enumerate(row_up):
                                if v in ["32", "NEW", "SAMPLE", "SPEC"]: v_col = i; break
                            
                            if d_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    row_data = df.iloc[d_idx]
                                    name = str(row_data[d_col]).strip().upper()
                                    val = parse_val(row_data[v_col])
                                    if len(name) > 3 and val > 0: specs[name] = val
                                break
        return {"specs": specs, "img": img_b}
    except: return None

# --- 3. GIAO DIỆN CHÍNH ---
st.title("🔍 AI Fashion Auditor V5.2")

with st.sidebar:
    st.header("📂 THƯ VIỆN MẪU")
    up_files = st.file_uploader("Nạp Techpack Gốc", accept_multiple_files=True)
    if up_files and st.button("🚀 LƯU VÀO KHO"):
        bar = st.progress(0)
        for i, f in enumerate(up_files):
            data = extract_pdf(f)
            if data and data['specs']:
                # Xử lý Vector 2D
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
                
                f_name = f.name.replace(".", "_")
                path = f"lib_{f_name}.png"
                supabase.storage.from_(BUCKET).upload(path=path, file=data['img'], file_options={"x-upsert":"true", "content-type":"image/png"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
            bar.progress((i + 1) / len(up_files))
        st.success("Đã nạp xong!")

# --- 4. ĐỐI SOÁT ---
audit_file = st.file_uploader("Tải file CẦN KIỂM TRA", type="pdf")
if audit_file:
    target = extract_pdf(audit_file)
    if target and target['specs']:
        st.write(f"✅ Đã đọc được {len(target['specs'])} thông số.")
        res = supabase.table("ai_data").select("*").execute()
        
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                # ÉP KIỂU 2D CHO VECTOR KIỂM TRA
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for item in res.data:
                if item.get('vector'):
                    # ÉP KIỂU 2D CHO VECTOR TRONG KHO
                    v_ref = np.array(item['vector']).reshape(1, -1)
                    # Tính toán Similarity an toàn
                    score = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": item, "sim": score})
            
            if matches:
                top = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
                st.subheader(f"✨ Khớp nhất: {top['data']['file_name']} ({top['sim']:.1f}%)")
                
                col1, col2 = st.columns(2)
                col1.image(target['img'], caption="File đang kiểm")
                col2.image(top['data']['image_url'], caption="Mẫu gốc trong kho")

                # So sánh bảng
                diffs = []
                ref_map = {clean_txt(k): v for k, v in top['data']['spec_json'].items()}
                for k, v in target['specs'].items():
                    v_ref = ref_map.get(clean_txt(k), 0)
                    d = round(v - v_ref, 3)
                    diffs.append({
                        "Hạng mục": k, "Đang kiểm": v, "Gốc": v_ref, 
                        "Lệch": d, "Kết quả": "🚩 Lệch" if abs(d) > 0.01 else "✔️ OK"
                    })
                
                st.table(pd.DataFrame(diffs))
