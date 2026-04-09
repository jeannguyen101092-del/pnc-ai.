import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay URL/KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V37.3", page_icon="📊")

# ================= INIT AI =================
@st.cache_resource
def load_ai():
    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feat = torch.nn.Sequential(*(list(base.children())[:-1])).eval()
    return feat

model_ai = load_ai()

def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
        return vec
    except: return None

# ================= CHỈ TẬP TRUNG POM =================
def extract_pom_only(pdf_file):
    data = {"img": None, "table": pd.DataFrame()}
    try:
        pdf_bytes = pdf_file.read()
        # 1. Lấy ảnh Sketch (Trang chứa hình vẽ)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            if any(k in page.get_text().upper() for k in ["SKETCH", "FRONT", "STYLE"]):
                data["img"] = page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                break
        if not data["img"]: data["img"] = doc.load_page(0).get_pixmap().tobytes("png")
        doc.close()

        # 2. Quét bảng POM
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all')
                    # Tìm bảng có cột Description/POM và các cột Size
                    for idx, row in df.iterrows():
                        row_up = [str(x).upper() for x in row if x]
                        if any(k in " ".join(row_up) for k in ["DESC", "POM", "MEASURE"]):
                            df.columns = [str(c).replace('\n',' ').strip().upper() for c in row]
                            data["table"] = df.iloc[idx+1:].reset_index(drop=True)
                            return data
        return data
    except: return None

# ================= APP LOGIC =================
supabase = create_client(URL, KEY)
try:
    res = supabase.table("ai_data").select("file_name, vector, spec_json").execute()
    samples = res.data if res.data else []
except: samples = []

with st.sidebar:
    st.header("📂 Kho dữ liệu POM")
    st.metric("Số mẫu trong kho", len(samples))
    files = st.file_uploader("Nạp Techpack mẫu (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Nạp/Cập nhật"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_pom_only(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                try:
                    supabase.storage.from_(BUCKET).upload(f"{name}.png", d['img'], {"content-type":"image/png","x-upsert":"true"})
                    vec = get_vector(d['img'])
                    specs = d['table'].to_json(orient='records')
                    supabase.table("ai_data").upsert({
                        "file_name": name, "vector": vec, "spec_json": specs, "image_url": supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    }).execute()
                except: pass
            p_bar.progress((i + 1) / len(files))
        st.success("Xong!"); st.rerun()

st.title("🔍 AI FASHION AUDITOR V37.3")
test_file = st.file_uploader("Upload PDF cần đối soát", type=["pdf"])

if test_file:
    test = extract_pom_only(test_file)
    if test and not test['table'].empty:
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(test['img'], caption="Sketch phát hiện được")
            t_vec = get_vector(test['img'])

        if samples and t_vec:
            # Tìm mẫu tương đồng nhất
            results = []
            for s in samples:
                try:
                    sim = float(cosine_similarity([t_vec], [s['vector']]))
                    results.append((s, sim))
                except: continue
            
            if results:
                results.sort(key=lambda x: x[1], reverse=True)
                ref, sim_val = results[0]
                
                with col2:
                    st.subheader(f"Mẫu khớp: {ref['file_name']} ({sim_val*100:.1f}%)")
                    st.progress(sim_val)

                    df_t = test['table']
                    # Lọc cột Size (Bỏ qua các cột thông tin phụ)
                    noise = ['DESC', 'POM', 'NO', 'TOL', 'ITEM', 'METHOD', 'COMMENT', 'UNNAMED']
                    size_cols = [c for c in df_t.columns if c and not any(n in str(c).upper() for n in noise)]
                    
                    sel_size = st.selectbox("🎯 Chọn Size đối soát:", size_cols)

                    try:
                        df_ref = pd.read_json(io.StringIO(ref['spec_json']))
                        audit = []
                        # Xác định cột Description làm Hạng mục
                        d_col = next((c for c in df_t.columns if any(k in str(c).upper() for k in ["DESC", "POM", "ITEM"])), df_t.columns[0])
                        
                        for _, row in df_t.iterrows():
                            desc = str(row[d_col]).strip()
                            if not desc or desc.upper() in ['NAN', 'NONE']: continue
                            
                            # Khớp Description giữa 2 bảng
                            match = df_ref[df_ref.iloc[:, 0].astype(str).str.upper().str.contains(desc[:8].upper(), na=False)]
                            if not match.empty:
                                try:
                                    v1 = float(re.findall(r"\d+\.?\d*", str(row[sel_size]))[0])
                                    v2 = float(re.findall(r"\d+\.?\d*", str(match.iloc[0].get(sel_size, '0')))[0])
                                    diff = round(v1 - v2, 2)
                                    audit.append({
                                        "Hạng mục (Description)": desc,
                                        "Thực tế": v1, "Mẫu gốc": v2, "Lệch": diff,
                                        "Kết quả": "✅ OK" if abs(diff)<=0.5 else "❌ LỆCH"
                                    })
                                except: continue
                        
                        if audit:
                            st.table(pd.DataFrame(audit))
                            # Nút xuất Excel
                            output = io.BytesIO()
                            pd.DataFrame(audit).to_excel(output, index=False)
                            st.download_button("📥 Tải báo cáo Excel", output.getvalue(), "POM_Audit.xlsx")
                    except: st.error("Dữ liệu mẫu trong kho không khớp cấu trúc POM.")
    else:
        st.error("Không tìm thấy bảng thông số (POM) trong file PDF này.")
