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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V49.0", page_icon="🔍")

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)
supabase = init_supabase()

@st.cache_resource
def load_model():
    # Sử dụng ResNet18 để trích xuất Vector (512 chiều)
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
        return vec.tolist() # Luôn trả về List 512 phần tử
    except:
        return None

# ================= PDF EXTRACTOR =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        # Đọc được cả phân số 1 1/2 và số thập phân
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
    
    # 1. Lấy ảnh trang 1 làm Thumbnail đối soát AI
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()

    # 2. Trích xuất bảng thông số
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if df.empty or len(df.columns) < 2: continue
                
                n_col, v_col = -1, -1
                for r_idx, row in df.iterrows():
                    row_up = [str(c).upper().strip() for c in row if c]
                    # Tìm cột POM Name/Description và cột Thông số (NEW/32/34...)
                    if any(x in " ".join(row_up) for x in ["POM", "DESCRIPTION"]):
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["POM", "DESC"]): n_col = i; break
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["NEW", "SAMPLE", "32", "34"]): v_col = i; break
                        
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
    if st.button("🧹 Dọn dẹp kho dữ liệu"):
        # Xóa sạch bản ghi cũ để nạp mới chuẩn Vector
        supabase.table("ai_data").delete().neq("id", 0).execute()
        st.success("Kho đã trống. Hãy nạp lại mẫu chuẩn!")
        st.rerun()

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
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data["specs"], "image_url": url
                }).execute()
            prog.progress((i + 1) / len(files))
        st.success("Nạp thành công!")
        st.rerun()

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V49.0")

# Lấy dữ liệu từ Database
db_res = supabase.table("ai_data").select("*").execute()
data_lib = db_res.data if db_res.data else []

if data_lib:
    st.info(f"Kho đang có **{len(data_lib)}** mẫu chuẩn. Sẵn sàng đối soát!")
else:
    st.warning("⚠️ Kho đang trống. Hãy nạp Techpack mẫu ở Sidebar.")

audit_file = st.file_uploader("Tải file cần đối soát (PDF)", type="pdf")

if audit_file:
    target = extract_pdf(audit_file)
    if target["specs"]:
        st.success(f"✅ Đã trích xuất {len(target['specs'])} thông số từ PDF.")
        v_test = get_vector(target["img"])
        
        if v_test and data_lib:
            v_test_np = np.array(v_test).reshape(1, -1)
            matches = []
            
            for item in data_lib:
                # Kiểm tra chặt chẽ độ dài vector (phải là 512)
                if item.get("vector") and len(item["vector"]) == 512:
                    v_ref = np.array(item["vector"]).reshape(1, -1)
                    score = float(cosine_similarity(v_test_np, v_ref))
                    matches.append({"item": item, "score": score})
            
            if matches:
                # Sắp xếp và lấy mẫu khớp nhất
                best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                
                st.subheader(f"✨ Kết quả khớp nhất: {best['item']['file_name']} ({best['score']*100:.1f}%)")
                
                col1, col2 = st.columns(2)
                col1.image(target["img"], caption="Bản vẽ đang kiểm", use_container_width=True)
                col2.image(best['item']['image_url'], caption="Mẫu gốc trong kho", use_container_width=True)

                # So khớp bảng POM
                ref_specs = best['item']['spec_json']
                diff_rows = []
                # Tạo map chuẩn hóa tên POM để so khớp
                clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): (k, v) for k, v in ref_specs.items()}

                for k, v in target["specs"].items():
                    k_clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                    ref_info = clean_ref.get(k_clean, (None, 0))
                    v_ref = ref_info[1]
                    diff = round(v - v_ref, 3)
                    
                    diff_rows.append({
                        "Hạng mục (POM)": k,
                        "Đang kiểm": v,
                        "Mẫu gốc": v_ref,
                        "Chênh lệch": diff,
                        "Kết quả": "✅ OK" if abs(diff) < 0.1 else "❌ FAIL"
                    })
                
                st.table(pd.DataFrame(diff_rows))
            else:
                st.error("❌ Lỗi so khớp: Vector trong kho không chuẩn. Hãy nhấn 'Dọn dẹp' và nạp lại Techpack mẫu.")
