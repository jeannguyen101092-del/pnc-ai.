# ✅ AI FASHION AUDITOR V36.1 FIXED (CHỐNG SẬP APP DO LỆCH VECTOR)
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V36.1", page_icon="📊")

# ================= INIT =================
@st.cache_resource
def init_supabase():
    try: return create_client(URL, KEY)
    except: return None

supabase = init_supabase()

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= VECTOR =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            base_vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()

        # Thêm histogram màu (64 bins) -> Tổng 2112 dims
        hist, _ = np.histogram(np.array(img).flatten(), bins=64, range=(0, 255))
        vec = np.concatenate([base_vec, hist.astype(np.float32)])
        return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()
    except:
        return None

# ================= PARSE POM TEXT =================
def parse_pom_text(text):
    lines = text.split("\n")
    data = []
    for line in lines:
        if re.search(r"\d+\.?\d*", line):
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 6:
                try:
                    data.append({
                        "DESC": parts[0],
                        "XS": float(parts[-5]), "S": float(parts[-4]),
                        "M": float(parts[-3]), "L": float(parts[-2]), "XL": float(parts[-1]),
                    })
                except: continue
    return pd.DataFrame(data)

# ================= EXTRACT PDF =================
def extract_techpack(pdf_file):
    data = {"img": None, "tables": []}
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text().upper()
            if any(k in text for k in ["SKETCH","DESIGN","DETAIL","CLOTHING"]):
                data["img"] = page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                break
        if data["img"] is None and len(doc) > 0:
            data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = (page.extract_text() or "")
                if "POM" in text.upper() or "SPECIFICATION" in text.upper():
                    df = parse_pom_text(text)
                    if not df.empty: data["tables"].append(df)
        return data
    except: return None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ================= LOAD DATA =================
samples = []
if supabase:
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res.data else []
    except: pass

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 Kho dữ liệu")
    files = st.file_uploader("Upload mẫu gốc (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Nạp/Cập nhật dữ liệu"):
        for f in files:
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                try:
                    supabase.storage.from_(BUCKET).upload(
                        path=f"{name}.png", file=d['img'],
                        file_options={"content-type":"image/png","x-upsert":"true"}
                    )
                    url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    vec = get_vector(d['img'])
                    specs = json.dumps([df.to_dict(orient='records') for df in d['tables']])
                    supabase.table("ai_data").upsert({
                        "file_name": name, "vector": vec, "spec_json": specs, "image_url": url
                    }).execute()
                except: pass
        st.success("Đã cập nhật kho dữ liệu mới!")
        st.rerun()

# ================= MAIN =================
st.title("🔍 AI FASHION AUDITOR V36.1 FIXED")
file = st.file_uploader("Upload file PDF cần kiểm tra", type=["pdf"])

if file:
    test = extract_techpack(file)
    if test and test['img']:
        col1, col2 = st.columns([1,1.3])
        with col1:
            st.image(test['img'], caption="Ảnh trích xuất")
            vec_test = get_vector(test['img'])

        if samples and vec_test:
            sims = []
            for s in samples:
                # FIX: Kiểm tra kích thước vector trước khi so sánh
                if s.get('vector') and len(s['vector']) == len(vec_test):
                    try:
                        sim = cosine_similarity([vec_test],[s['vector']])[0][0]
                        sims.append((s, sim))
                    except: continue
            
            if not sims:
                st.error("Kho dữ liệu cũ không tương thích. Vui lòng nhấn 'Nạp' lại dữ liệu ở Sidebar.")
            else:
                sims.sort(key=lambda x: x[1], reverse=True)
                with col2:
                    names = [s[0]['file_name'] for s in sims]
                    sel = st.selectbox("Chọn mẫu so sánh", names)
                    ref = next(s[0] for s in sims if s[0]['file_name']==sel)

                    sim_val = next(s[1] for s in sims if s[0]['file_name']==sel)
                    st.metric("Độ tương đồng AI", f"{sim_val*100:.2f}%")
                    st.progress(float(sim_val))

                    size = st.selectbox("Chọn size đối soát", ["XS","S","M","L","XL"])
                    if test['tables'] and ref['spec_json']:
                        df_test = test['tables'][0]
                        ref_specs = json.loads(ref['spec_json'])
                        if ref_specs:
                            df_ref = pd.DataFrame(ref_specs[0])
                            results = []
                            for i in range(min(len(df_test), len(df_ref))):
                                try:
                                    v1 = df_test.iloc[i][size]
                                    v2 = df_ref.iloc[i][size]
                                    diff = round(float(v1) - float(v2), 2)
                                    results.append({
                                        "Hạng mục": df_test.iloc[i]['DESC'],
                                        "Thực tế": v1, "Mẫu gốc": v2, "Lệch": diff,
                                        "Kết quả": "✅ OK" if abs(diff)<=0.5 else "❌ LỆCH"
                                    })
                                except: continue
                            if results:
                                res_df = pd.DataFrame(results)
                                st.table(res_df)
                                st.download_button("📥 Xuất Excel", to_excel(res_df), "report.xlsx")
    else:
        st.error("Không thể đọc được ảnh hoặc bảng từ PDF này.")
