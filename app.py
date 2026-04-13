import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- 1. CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V5.4")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# --- 2. HELPERS ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null', '0']: return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0] # Lấy phần tử đầu tiên của list match
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_txt(t): return re.sub(r'[^A-Z0-9]', '', str(t).upper())

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

# --- 3. MAIN ---
st.title("🔍 AI Fashion Auditor V5.4")

with st.sidebar:
    st.header("📂 THƯ VIỆN MẪU")
    up_files = st.file_uploader("Nạp Techpack Gốc", accept_multiple_files=True)
    if up_files and st.button("🚀 LƯU VÀO KHO"):
        bar = st.progress(0)
        for i, f in enumerate(up_files):
            data = extract_pdf(f)
            if data and data['specs']:
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().tolist()
                
                f_name = f.name.replace(".", "_")
                path = f"lib_{f_name}.png"
                supabase.storage.from_(BUCKET).upload(path=path, file=data['img'], file_options={"x-upsert":"true", "content-type":"image/png"})
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
            bar.progress((i + 1) / len(up_files))
        st.success("Đã nạp xong!")

audit_file = st.file_uploader("Tải file CẦN KIỂM TRA", type="pdf")
if audit_file:
    target = extract_pdf(audit_file)
    if target and target['specs']:
        st.success(f"✅ Đã đọc được {len(target['specs'])} thông số.")
        res = supabase.table("ai_data").select("*").execute()
        
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                # ÉP KIỂU 2D MẠNH BẰNG NP.ATLEAST_2D
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy()
                v_test = np.atleast_2d(v_test)

            matches = []
            for item in res.data:
                if item.get('vector'):
                    v_ref = np.array(item['vector'])
                    v_ref = np.atleast_2d(v_ref)
                    
                    # Tính toán an toàn, lấy giá trị scalar [0][0]
                    sim_matrix = cosine_similarity(v_test, v_ref)
                    score = float(sim_matrix[0][0]) * 100
                    matches.append({"data": item, "sim": score})
            
            if matches:
                top_matches = sorted(matches, key=lambda x: x['sim'], reverse=True)
                top = top_matches[0] # Lấy mẫu cao điểm nhất
                
                st.subheader(f"✨ Khớp nhất: {top['data']['file_name']} ({top['sim']:.1f}%)")
                
                col1, col2 = st.columns(2)
                col1.image(target['img'], caption="File đang kiểm")
                col2.image(top['data']['image_url'], caption="Mẫu gốc")

                diffs = []
                ref_map = {clean_txt(k): v for k, v in top['data']['spec_json'].items()}
                for k, v in target['specs'].items():
                    v_ref = ref_map.get(clean_txt(k), 0)
                    d = round(v - v_ref, 3)
                    diffs.append({"Hạng mục": k, "Đang kiểm": v, "Gốc": v_ref, "Lệch": d, "Kết quả": "🚩 Lệch" if abs(d) > 0.01 else "✔️ OK"})
                st.table(pd.DataFrame(diffs))
