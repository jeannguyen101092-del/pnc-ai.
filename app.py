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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V48.7", page_icon="🔍")

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)
supabase = init_supabase()

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
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= EXTRACT PDF =================
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
                name_col, val_col = -1, -1
                for r_idx, row in df.iterrows():
                    row_str = [str(c).upper().strip() for c in row if c is not None]
                    if any(x in " ".join(row_str) for x in ["POM", "DESCRIPTION"]):
                        for i, v in enumerate(row_str):
                            if "POM" in v or "DESC" in v: name_col = i; break
                        for i, v in enumerate(row_str):
                            if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32", "34", "36"]):
                                val_col = i; break
                        if name_col != -1 and val_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                p_name = str(d_row[name_col]).strip().upper()
                                if len(p_name) < 3: continue
                                val = parse_val(d_row[val_col])
                                if val > 0: specs[p_name] = val
                            break
    return {"specs": specs, "img": img_bytes}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ CÀI ĐẶT")
    if st.button("🧹 Dọn dẹp mẫu lỗi"):
        db_all = supabase.table("ai_data").select("id, vector").execute()
        for item in db_all.data:
            if not item.get("vector") or len(item["vector"]) != 512:
                supabase.table("ai_data").delete().eq("id", item["id"]).execute()
        st.success("Đã dọn dẹp xong!")
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
st.title("🔍 AI Fashion Auditor V48.7")

# KIỂM TRA TRẠNG THÁI DB
db_res = supabase.table("ai_data").select("*").execute()
data_lib = db_res.data if db_res.data else []

if not data_lib:
    st.warning("⚠️ Kho dữ liệu đang trống. Hãy nạp mẫu ở Sidebar bên trái.")
else:
    audit_file = st.file_uploader("Tải lên file cần đối soát (PDF)", type="pdf")
    if audit_file:
        target = extract_pdf(audit_file)
        if target["specs"]:
            st.success(f"✅ Đã trích xuất {len(target['specs'])} thông số.")
            
            v_test = get_vector(target["img"])
            if v_test:
                v_test_np = np.array(v_test).reshape(1, -1)
                matches = []
                for item in data_lib:
                    # FIX: Kiểm tra kỹ định dạng vector
                    if item.get("vector") and len(item["vector"]) == 512:
                        v_ref = np.array(item["vector"]).reshape(1, -1)
                        score = float(cosine_similarity(v_test_np, v_ref))
                        matches.append({"item": item, "score": score})
                
                if matches:
                    # Lấy mẫu cao điểm nhất
                    best_list = sorted(matches, key=lambda x: x['score'], reverse=True)
                    best = best_list[0] # Lấy phần tử đầu tiên
                    
                    st.subheader(f"✨ Khớp nhất: {best['item']['file_name']} ({best['score']*100:.1f}%)")
                    
                    c1, c2 = st.columns(2)
                    c1.image(target["img"], caption="Bản đang kiểm")
                    c2.image(best['item']['image_url'], caption="Mẫu gốc trong kho")

                    # HIỂN THỊ BẢNG SO SÁNH
                    ref_specs = best['item']['spec_json']
                    diff_rows = []
                    for k, v in target["specs"].items():
                        k_clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                        v_ref = 0
                        for r_k, r_v in ref_specs.items():
                            if re.sub(r'[^A-Z0-9]', '', r_k.upper()) == k_clean:
                                v_ref = r_v; break
                        diff = round(v - v_ref, 3)
                        diff_rows.append({"Hạng mục": k, "Đang kiểm": v, "Gốc": v_ref, "Lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.1 else "❌ FAIL"})
                    st.table(pd.DataFrame(diff_rows))
                else:
                    st.error("❌ Lỗi so khớp: Các mẫu trong kho có Vector không hợp lệ (không phải 512 chiều). Hãy nhấn 'Dọn dẹp mẫu lỗi' và nạp lại Techpack mẫu.")
