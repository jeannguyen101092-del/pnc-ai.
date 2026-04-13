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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V48.1", page_icon="🔥")

# Khởi tạo Supabase an toàn
@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)
supabase = init_supabase()

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= VECTOR =================
def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
        return vec.tolist()
    except:
        return None

# ================= UTILS =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
        # Hỗ trợ phân số như 1 1/2 hoặc 23.75
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except:
        return 0

def clean_key(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT NÂNG CẤP =================
def extract_pdf(file):
    specs, img_bytes = {}, None
    file.seek(0)
    pdf_content = file.read()

    # 1. Lấy ảnh trang đầu
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Quét bảng thông số thông minh
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if df.empty or len(df.columns) < 2: continue

                # Tìm dòng tiêu đề có chứa POM/Description và Sample/New
                name_col, val_col = -1, -1
                for r_idx, row in df.iterrows():
                    row_str = [str(c).upper() for c in row if c]
                    if any(x in " ".join(row_str) for x in ["POM", "DESCRIPTION"]):
                        name_col = next((i for i, v in enumerate(row_str) if "POM" in v or "DESC" in v), 0)
                        # Tìm cột chứa giá trị (ưu tiên Sample Size hoặc NEW)
                        for i, v in enumerate(row_str):
                            if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32"]):
                                val_col = i; break
                        
                        if name_col != -1 and val_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                p_name = str(d_row[name_col]).strip().upper()
                                if len(p_name) < 3 or "TOL" in p_name: continue
                                val = parse_val(d_row[val_col])
                                if val > 0: specs[p_name] = val
                            break
    return {"specs": specs, "img": img_bytes}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ SETTINGS")
    
    try:
        count_res = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("📦 Số mẫu trong kho", count_res.count if count_res.count else 0)
    except: st.warning("Chưa kết nối DB")

    files = st.file_uploader("Upload Techpack Mẫu", accept_multiple_files=True)
    if files and st.button("🚀 Nạp vào kho"):
        for f in files:
            data = extract_pdf(f)
            if not data["specs"]:
                st.error(f"Không tìm thấy bảng thông số trong: {f.name}")
                continue
            
            vec = get_vector(data["img"])
            path = f"lib_{f.name.replace('.pdf','')}.png"
            supabase.storage.from_(BUCKET).upload(path, data["img"], {"upsert":"true", "content-type":"image/png"})
            url = supabase.storage.from_(BUCKET).get_public_url(path)

            supabase.table("ai_data").insert({
                "file_name": f.name, "vector": vec, "spec_json": data["specs"], "image_url": url
            }).execute()
            st.success(f"Đã nạp: {f.name}")
        st.rerun()

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V48.1")

# HÀM HIỂN THỊ THƯ VIỆN
db_res = supabase.table("ai_data").select("*").execute()
data_lib = db_res.data if db_res.data else []

if data_lib:
    with st.expander("📦 THƯ VIỆN DỮ LIỆU (DATA LIBRARY)", expanded=False):
        cols = st.columns(5)
        for idx, item in enumerate(data_lib):
            with cols[idx % 5]:
                st.image(item["image_url"], caption=item["file_name"], use_container_width=True)

# ================= KIỂM TRA =================
st.subheader("🔎 KIỂM TRA FILE MỚI")
audit_file = st.file_uploader("Upload file đối soát (PDF)", type="pdf", key="auditor")

if audit_file:
    target = extract_pdf(audit_file)
    if not target["specs"]:
        st.error("❌ Không thể trích xuất bảng thông số từ file này.")
    else:
        v_test = get_vector(target["img"])
        if v_test:
            v_test_np = np.array(v_test).reshape(1, -1)
            matches = []
            
            for item in data_lib:
                if item.get("vector") and len(item["vector"]) == 512:
                    v_ref = np.array(item["vector"]).reshape(1, -1)
                    score = float(cosine_similarity(v_test_np, v_ref)[0][0])
                    matches.append({"item": item, "score": score})
            
            if not matches:
                st.warning("⚠️ Không có dữ liệu mẫu hợp lệ trong kho để so sánh.")
            else:
                best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                st.subheader(f"✨ Kết quả khớp nhất: {best['item']['file_name']} ({best['score']*100:.1f}%)")
                
                # So sánh bảng
                col_img1, col_img2 = st.columns(2)
                col_img1.image(target["img"], caption="File đang kiểm")
                col_img2.image(best["item"]["image_url"], caption="Mẫu gốc đối chiếu")

                diff_rows = []
                ref_specs = best["item"]["spec_json"]
                ref_map = {clean_key(k): v for k, v in ref_specs.items()}

                for k, v in target["specs"].items():
                    v_ref = ref_map.get(clean_key(k), 0)
                    diff = round(v - v_ref, 3)
                    diff_rows.append({
                        "Hạng mục (POM)": k,
                        "Đang kiểm": v,
                        "Mẫu gốc": v_ref,
                        "Chênh lệch": diff,
                        "Kết quả": "✅ OK" if abs(diff) < 0.1 else "❌ FAIL"
                    })
                
                df_res = pd.DataFrame(diff_rows)
                st.table(df_res.style.applymap(lambda x: 'color: red' if x == "❌ FAIL" else '', subset=['Kết quả']))
