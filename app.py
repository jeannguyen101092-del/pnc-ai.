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

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V36.2", page_icon="📊")

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
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            base_vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
        hist, _ = np.histogram(np.array(img).flatten(), bins=64, range=(0, 255))
        vec = np.concatenate([base_vec, hist.astype(np.float32)])
        return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()
    except: return None

# ================= NEW: SMART TABLE EXTRACT =================
def clean_table(df):
    """Làm sạch bảng: tìm header và loại bỏ rác"""
    for idx, row in df.iterrows():
        row_str = row.astype(str).str.upper()
        # Tìm dòng chứa các Size hoặc Description
        if any(k in "".join(row_str.values) for k in ["DESC", "POM", "SIZE", " TOL "]):
            df.columns = row_str.values
            return df.iloc[idx+1:].reset_index(drop=True)
    return None

def extract_techpack(pdf_file):
    data = {"img": None, "tables": []}
    try:
        pdf_bytes = pdf_file.read()
        # 1. Trích xuất ảnh
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            if any(k in page.get_text().upper() for k in ["SKETCH","DESIGN","DETAIL","STYLE"]):
                data["img"] = page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                break
        if data["img"] is None and len(doc) > 0:
            data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        # 2. Trích xuất bảng (Dùng pdfplumber table thay vì text)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table)
                    cleaned_df = clean_table(df)
                    if cleaned_df is not None and not cleaned_df.empty:
                        # Chuẩn hóa cột DESC
                        if cleaned_df.columns[0] == 'NONE' or cleaned_df.columns[0] == '':
                             cleaned_df.rename(columns={cleaned_df.columns[0]: "DESC"}, inplace=True)
                        data["tables"].append(cleaned_df)
        return data
    except: return None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ================= LOAD SAMPLES =================
samples = []
if supabase:
    try:
        res = supabase.table("ai_data").select("*").execute()
        samples = res.data if res.data else []
    except: pass

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 Kho dữ liệu")
    files = st.file_uploader("Upload mẫu gốc", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Cập nhật kho dữ liệu"):
        for f in files:
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                try:
                    supabase.storage.from_(BUCKET).upload(f"{name}.png", d['img'], {"content-type":"image/png","x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    vec = get_vector(d['img'])
                    specs = json.dumps([df.to_dict(orient='records') for df in d['tables']])
                    supabase.table("ai_data").upsert({"file_name": name, "vector": vec, "spec_json": specs, "image_url": url}).execute()
                except: pass
        st.success("Đã nạp xong!"); st.rerun()

# ================= MAIN =================
st.title("🔍 AI FASHION AUDITOR V36.2")
file = st.file_uploader("Upload PDF kiểm tra", type=["pdf"])

if file:
    test = extract_techpack(file)
    if test and test['img']:
        col1, col2 = st.columns([1,1.3])
        with col1:
            st.image(test['img'])
            vec_test = get_vector(test['img'])

        if samples and vec_test:
            sims = []
            for s in samples:
                if s.get('vector') and len(s['vector']) == len(vec_test):
                    sim = float(cosine_similarity([vec_test],[s['vector']])[0][0])
                    sims.append((s, sim))
            
            if sims:
                sims.sort(key=lambda x: x[1], reverse=True)
                with col2:
                    ref = sims[0][0]
                    st.metric("Mẫu khớp nhất", ref['file_name'], f"{sims[0][1]*100:.1f}%")
                    
                    # Lấy danh sách size từ bảng
                    if test['tables']:
                        df_test = test['tables'][0]
                        # Tìm các cột có tên ngắn (thường là size)
                        size_options = [c for c in df_test.columns if len(str(c)) <= 3 and str(c).upper() not in ["TOL", "NO"]]
                        size = st.selectbox("Chọn size đối soát", size_options if size_options else ["M"])

                        try:
                            df_ref = pd.DataFrame(json.loads(ref['spec_json'])[0])
                            results = []
                            for i, row in df_test.iterrows():
                                pom_desc = str(row.values[0])
                                # Tìm dòng tương ứng ở mẫu gốc bằng tên POM
                                match = df_ref[df_ref.iloc[:, 0].astype(str).str.contains(pom_desc[:10], case=False, na=False)]
                                if not match.empty:
                                    v1, v2 = row[size], match.iloc[0][size]
                                    diff = round(float(v1) - float(v2), 2)
                                    results.append({"POM": pom_desc, "Thực tế": v1, "Gốc": v2, "Lệch": diff, "Kết quả": "✅" if abs(diff)<=0.5 else "❌"})
                            
                            if results:
                                res_df = pd.DataFrame(results)
                                st.table(res_df)
                                st.download_button("📥 Xuất Excel", to_excel(res_df), "report.xlsx")
                            else: st.warning("Không khớp được dòng thông số.")
                        except: st.error("Lỗi cấu hình bảng gốc.")
    else: st.error("PDF này không có dữ liệu bảng hoặc ảnh.")
