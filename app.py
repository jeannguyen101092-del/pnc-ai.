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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V48.4", page_icon="🔍")

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)
supabase = init_supabase()

# ================= MODEL AI =================
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
    except Exception as e:
        st.error(f"Lỗi tạo Vector: {e}")
        return None

# ================= TRÍCH XUẤT PDF (FIX LỖI NHẬN DIỆN BẢNG) =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
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
    
    # 1. Chụp ảnh trang 1
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Quét bảng (Nâng cấp khả năng quét)
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if df.empty or len(df.columns) < 2: continue
                
                # Tìm tiêu đề cột thông minh hơn
                name_col, val_col = -1, -1
                for r_idx, row in df.iterrows():
                    row_str = [str(c).upper() for c in row if c]
                    # Nếu dòng chứa "DESCRIPTION" hoặc "POM"
                    if any(x in " ".join(row_str) for x in ["POM", "DESCRIPTION", "DIMENSION"]):
                        name_col = next((i for i, v in enumerate(row_str) if any(x in v for x in ["POM", "DESC", "DIMENSION"])), 0)
                        # Tìm cột giá trị (Ưu tiên cột số 32, 34, NEW hoặc cột cuối cùng không phải Tol)
                        for i, v in enumerate(row_str):
                            if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32", "34", "36"]):
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
    st.header("⚙️ CÀI ĐẶT")
    if st.button("🧹 Dọn dẹp mẫu lỗi"):
        db_all = supabase.table("ai_data").select("id, vector").execute()
        count_del = 0
        for item in db_all.data:
            if not item.get("vector") or len(item["vector"]) != 512:
                supabase.table("ai_data").delete().eq("id", item["id"]).execute()
                count_del += 1
        st.success(f"Đã dọn dẹp {count_del} mẫu lỗi.")
        st.rerun()

    files = st.file_uploader("Upload Techpack Mẫu", accept_multiple_files=True)
    if files and st.button("🚀 Nạp vào kho"):
        for f in files:
            data = extract_pdf(f)
            vec = get_vector(data["img"])
            if data["specs"] and vec:
                path = f"lib_{f.name.replace(' ', '_')}.png"
                supabase.storage.from_(BUCKET).upload(path, data["img"], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data["specs"], "image_url": url
                }).execute()
        st.success("Nạp thành công!")
        st.rerun()

# ================= PHẦN CHÍNH =================
st.title("🔍 AI Fashion Auditor V48.4")

# Lấy dữ liệu Database
db_res = supabase.table("ai_data").select("*").execute()
data_lib = db_res.data if db_res.data else []

# Hiển thị trạng thái kho
with st.expander("📦 Trạng thái kho dữ liệu"):
    if not data_lib:
        st.error("Kho đang trống! Hãy nạp mẫu ở Sidebar.")
    else:
        st.write(f"Hiện có {len(data_lib)} mẫu chuẩn trong kho.")

# PHẦN ĐỐI SOÁT
audit_file = st.file_uploader("Tải lên file cần đối soát (PDF)", type="pdf")
if audit_file:
    target = extract_pdf(audit_file)
    
    if not target["specs"]:
        st.error("❌ Không tìm thấy bảng thông số trong PDF. Hãy kiểm tra lại file của bạn.")
    else:
        st.success(f"✅ Đã trích xuất {len(target['specs'])} thông số.")
        
        v_test = get_vector(target["img"])
        if v_test and data_lib:
            v_test_np = np.atleast_2d(v_test)
            matches = []
            for item in data_lib:
                if item.get("vector") and len(item["vector"]) == 512:
                    v_ref = np.atleast_2d(item["vector"])
                    score = float(cosine_similarity(v_test_np, v_ref))
                    matches.append({"item": item, "score": score})
            
            if matches:
                best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                st.subheader(f"✨ Khớp nhất: {best['item']['file_name']} ({best['score']*100:.1f}%)")
                
                # Hiển thị ảnh và bảng so sánh
                c1, c2 = st.columns(2)
                c1.image(target["img"], caption="Bản đang kiểm")
                c2.image(best['item']['image_url'], caption="Bản mẫu gốc")

                # Logic so sánh POM...
                diff_rows = []
                ref_specs = best["item"]["spec_json"]
                for k, v in target["specs"].items():
                    # Tìm thông số khớp tên trong kho
                    k_clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                    v_ref = 0
                    for r_k, r_v in ref_specs.items():
                        if re.sub(r'[^A-Z0-9]', '', r_k.upper()) == k_clean:
                            v_ref = r_v
                            break
                    
                    diff = round(v - v_ref, 3)
                    diff_rows.append({
                        "Hạng mục": k, "Đang kiểm": v, "Mẫu gốc": v_ref, 
                        "Chênh lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.1 else "❌ FAIL"
                    })
                st.table(pd.DataFrame(diff_rows))
            else:
                st.warning("Không tìm thấy mẫu tương ứng trong kho.")
