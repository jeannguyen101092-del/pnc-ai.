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

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V36.4", page_icon="📊")

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

# ================= VECTOR (Fix lỗi kích thước) =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            base_vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
        # Thêm Histogram 64 bins
        hist, _ = np.histogram(np.array(img).flatten(), bins=64, range=(0, 255))
        vec = np.concatenate([base_vec, hist.astype(np.float32)])
        return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()
    except: return None

# ================= TRÍCH XUẤT PDF (Fix cho mẫu Express) =================
def extract_techpack(pdf_file):
    data = {"img": None, "tables": pd.DataFrame()}
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Tìm ảnh Sketch
        for i in range(len(doc)):
            page = doc.load_page(i)
            if any(k in page.get_text().upper() for k in ["SKETCH","FRONT","BACK","STYLE"]):
                data["img"] = page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                break
        if not data["img"] and len(doc)>0:
            data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        # Tìm bảng thông số
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            tables = []
            for page in pdf.pages:
                extracted = page.extract_tables()
                for e in extracted:
                    df = pd.DataFrame(e).dropna(how='all')
                    if len(df.columns) >= 3: tables.append(df)
            
            if tables:
                # Lấy bảng dài nhất (thường là bảng POM)
                main_df = max(tables, key=len)
                # Tìm dòng header (dòng chứa chữ hoặc size)
                for idx, row in main_df.iterrows():
                    if any(str(x).strip().upper() in ["S","M","L","XL","DESC","POM"] for x in row):
                        main_df.columns = [str(c).replace('\n',' ') for c in row]
                        data["tables"] = main_df.iloc[idx+1:].reset_index(drop=True)
                        break
        return data
    except: return None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# ================= DATA =================
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
    if files and st.button("🚀 Cập nhật kho"):
        for f in files:
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                try:
                    supabase.storage.from_(BUCKET).upload(f"{name}.png", d['img'], {"content-type":"image/png","x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    vec = get_vector(d['img'])
                    # Lưu bảng dưới dạng JSON
                    specs = d['tables'].to_json(orient='records')
                    supabase.table("ai_data").upsert({"file_name": name, "vector": vec, "spec_json": specs, "image_url": url}).execute()
                except: pass
        st.success("Đã nạp xong!"); st.rerun()

# ================= MAIN =================
st.title("🔍 AI FASHION AUDITOR V36.4")
file = st.file_uploader("Upload PDF kiểm tra", type=["pdf"])

if file:
    test = extract_techpack(file)
    if test and test['img']:
        col1, col2 = st.columns([1,1.3])
        with col1:
            st.image(test['img'], caption="Bản vẽ phát hiện được")
            vec_test = get_vector(test['img'])

        if samples and vec_test:
            # So sánh AI (Fix TypeError bằng try-except)
            results = []
            for s in samples:
                try:
                    if s.get('vector') and len(s['vector']) == len(vec_test):
                        sim = float(cosine_similarity([vec_test],[s['vector']])[0][0])
                        results.append((s, sim))
                except: continue
            
            if results:
                results.sort(key=lambda x: x[1], reverse=True)
                with col2:
                    ref, sim_best = results[0]
                    st.metric("Mẫu khớp nhất", ref['file_name'], f"{sim_best*100:.1f}%")
                    st.progress(sim_best)

                    df_test = test['tables']
                    if not df_test.empty:
                        # Lấy danh sách size từ header bảng
                        size_cols = [c for c in df_test.columns if c and str(c).strip()]
                        sel_size = st.selectbox("Chọn cột thông số (Size):", size_cols)

                        try:
                            # Load bảng gốc
                            df_ref = pd.read_json(io.StringIO(ref['spec_json']))
                            audit = []
                            for _, row in df_test.iterrows():
                                desc = str(row.iloc[0]).strip().upper() # Cột đầu là Description
                                if not desc or desc == 'NAN': continue
                                
                                # Tìm dòng tương ứng bằng Description
                                match = df_ref[df_ref.iloc[:, 0].astype(str).str.upper().str.contains(desc[:10], na=False)]
                                if not match.empty:
                                    try:
                                        v1 = re.findall(r"\d+\.?\d*", str(row[sel_size]))[0]
                                        v2 = re.findall(r"\d+\.?\d*", str(match.iloc[0][sel_size]))[0]
                                        diff = round(float(v1) - float(v2), 2)
                                        audit.append({"Hạng mục": desc, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": diff, "Kquả": "✅" if abs(diff)<=0.5 else "❌"})
                                    except: continue
                            
                            if audit:
                                res_df = pd.DataFrame(audit)
                                st.table(res_df)
                                st.download_button("📥 Xuất Excel", to_excel(res_df), "report.xlsx")
                            else: st.warning("Không khớp được dữ liệu giữa 2 bảng.")
                        except: st.error("Lỗi: Cấu trúc bảng mẫu gốc trong kho không khớp.")
            else: st.error("⚠️ Kho dữ liệu cũ không tương thích. Vui lòng 'Cập nhật kho' ở bên trái.")
    else: st.error("Không tìm thấy ảnh hoặc bảng trong PDF.")
