import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V53", page_icon="🔍")

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)
supabase = init_supabase()

@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= VECTOR PROCESSOR =================
def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
        return vec.astype(np.float32).tolist() 
    except:
        return None

# ================= PDF EXTRACTOR =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_pdf(file):
    specs, img_bytes = {}, None
    file.seek(0)
    pdf_content = file.read()
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if df.empty or len(df.columns) < 2: continue
                n_col, v_col = -1, -1
                for r_idx, row in df.iterrows():
                    row_up = [str(c).upper().strip() for c in row if c]
                    if any(x in " ".join(row_up) for x in ["POM", "DESCRIPTION", "DIMENSION"]):
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["POM", "DESC", "DIMENSION"]): n_col = i; break
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["NEW", "SAMPLE", "SPEC", "32", "34"]): v_col = i; break
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[n_col]).strip().upper()
                                val = parse_val(d_row[v_col])
                                if len(name) > 3 and val > 0: specs[name] = val
                            break
    return {"specs": specs, "img": img_bytes}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ CÀI ĐẶT")
    
    # FIX: Lệnh xóa an toàn hơn với try-except
    if st.button("🧹 Dọn dẹp kho dữ liệu"):
        try:
            # Sử dụng filter lọc cột id không rỗng để xóa toàn bộ
            supabase.table("ai_data").delete().filter("id", "not.is", "null").execute()
            st.success("Kho đã trống. Hãy nạp lại mẫu chuẩn!")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi: Có thể bạn chưa cấp quyền DELETE trên Supabase. ({e})")

    files = st.file_uploader("Upload Techpack Mẫu", accept_multiple_files=True)
    if files and st.button("🚀 Nạp vào kho"):
        prog = st.progress(0)
        for i, f in enumerate(files):
            data = extract_pdf(f)
            vec = get_vector(data["img"])
            if data["specs"] and vec:
                path = f"lib_{f.name.replace(' ', '_')}.png"
                supabase.storage.from_(BUCKET).upload(path, data["img"], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data["specs"], "image_url": url}).execute()
            prog.progress((i + 1) / len(files))
        st.success("Nạp thành công!")
        st.rerun()

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V53")

try:
    db_res = supabase.table("ai_data").select("*").execute()
    data_lib = db_res.data if db_res.data else []
except:
    data_lib = []

if data_lib:
    st.info(f"Kho đang có **{len(data_lib)}** mẫu chuẩn.")
else:
    st.warning("⚠️ Kho đang trống.")

audit_file = st.file_uploader("Tải file cần đối soát (PDF)", type="pdf")
if audit_file:
    target = extract_pdf(audit_file)
    if target["specs"]:
        v_test = get_vector(target["img"])
        if v_test and data_lib:
            v_test_np = np.atleast_2d(v_test).astype(np.float32)
            matches = []
            for item in data_lib:
                try:
                    v_ref = np.atleast_2d(item["vector"]).astype(np.float32)
                    score = float(cosine_similarity(v_test_np, v_ref))
                    matches.append({"item": item, "score": score})
                except: continue
            
            if matches:
                best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                st.subheader(f"✨ Khớp nhất: {best['item']['file_name']} ({best['score']*100:.1f}%)")
                
                c1, c2 = st.columns(2)
                c1.image(target["img"], caption="Bản đang kiểm")
                c2.image(best['item']['image_url'], caption="Mẫu gốc")

                diff_rows = []
                clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): (k, v) for k, v in best['item']['spec_json'].items()}
                for k, v in target["specs"].items():
                    k_clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                    v_ref = clean_ref.get(k_clean, (None, 0))[1]
                    diff = round(v - v_ref, 3)
                    diff_rows.append({"Hạng mục": k, "Đang kiểm": v, "Gốc": v_ref, "Lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.1 else "❌ FAIL"})
                st.table(pd.DataFrame(diff_rows))
