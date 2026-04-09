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

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V36.3", page_icon="📊")

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

# ================= CẢI TIẾN TRÍCH XUẤT =================
def extract_techpack(pdf_file):
    data = {"img": None, "tables": []}
    try:
        pdf_bytes = pdf_file.read()
        
        # 1. Trích xuất ảnh (Ưu tiên trang Sketch, nếu ko thấy lấy Page 1)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            if any(k in page.get_text().upper() for k in ["SKETCH","DESIGN","DETAIL","STYLE","CLOTHING"]):
                data["img"] = page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                break
        if data["img"] is None and len(doc) > 0:
            data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        # 2. Trích xuất bảng (Lấy tất cả các bảng lớn)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_tables = []
            for page in pdf.pages:
                tbs = page.extract_tables()
                if tbs:
                    for tb in tbs:
                        df = pd.DataFrame(tb)
                        if len(df.columns) > 3 and len(df) > 2: # Lọc bỏ các bảng quá nhỏ
                            all_tables.append(df)
            
            # Gán header cho bảng lớn nhất tìm được
            if all_tables:
                main_df = max(all_tables, key=len)
                # Tự động tìm dòng header (dòng nào nhiều chữ nhất)
                main_df.columns = main_df.iloc[0]
                data["tables"] = main_df.iloc[1:].reset_index(drop=True)
                
        return data
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

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
    st.header("📦 Kho dữ liệu")
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
                    # Lưu bảng chính
                    specs = d['tables'].to_json(orient='records') if not isinstance(d['tables'], list) else "{}"
                    supabase.table("ai_data").upsert({"file_name": name, "vector": vec, "spec_json": specs, "image_url": url}).execute()
                except Exception as e: st.error(e)
        st.success("Xong!"); st.rerun()

# ================= MAIN =================
st.title("🔍 AI FASHION AUDITOR V36.3")
file = st.file_uploader("Upload PDF kiểm tra", type=["pdf"])

if file:
    test = extract_techpack(file)
    if test and test['img']:
        col1, col2 = st.columns([1,1.3])
        with col1:
            st.image(test['img'], caption="Bản vẽ phát hiện được")
            vec_test = get_vector(test['img'])

        if samples and vec_test:
            # So sánh AI
            results = []
            for s in samples:
                if s.get('vector') and len(s['vector']) == len(vec_test):
                    sim = float(cosine_similarity([vec_test],[s['vector']]))
                    results.append((s, sim))
            
            if results:
                results.sort(key=lambda x: x[1], reverse=True)
                with col2:
                    ref, sim_best = results[0]
                    st.metric("Mẫu khớp nhất", ref['file_name'], f"{sim_best*100:.1f}%")
                    st.progress(sim_best)

                    df_test = test['tables']
                    if not df_test.empty:
                        # Tự tìm các cột Size (cột có dữ liệu số)
                        size_options = [c for c in df_test.columns if c and str(c).strip()]
                        size = st.selectbox("Chọn cột thông số (Size):", size_options)

                        try:
                            df_ref = pd.read_json(io.StringIO(ref['spec_json']))
                            audit = []
                            for _, row in df_test.iterrows():
                                desc = str(row.iloc[0]) # Cột đầu thường là Description
                                match = df_ref[df_ref.iloc[:, 0].astype(str).str.contains(desc[:8], case=False, na=False)]
                                if not match.empty:
                                    v1 = re.findall(r"\d+\.?\d*", str(row[size]))
                                    v2 = re.findall(r"\d+\.?\d*", str(match.iloc[0][size]))
                                    if v1 and v2:
                                        diff = round(float(v1[0]) - float(v2[0]), 2)
                                        audit.append({"Hạng mục": desc, "Thực tế": v1[0], "Mẫu gốc": v2[0], "Lệch": diff, "Kết quả": "✅" if abs(diff)<=0.5 else "❌"})
                            
                            if audit:
                                res_df = pd.DataFrame(audit)
                                st.table(res_df)
                                st.download_button("📥 Excel", to_excel(res_df), "report.xlsx")
                        except: st.info("Không khớp được cấu hình bảng cũ.")
    else:
        st.error("⚠️ Không tìm thấy bảng dữ liệu. Kiểm tra xem PDF có phải dạng ảnh scan không?")
