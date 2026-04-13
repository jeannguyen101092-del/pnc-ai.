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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V48.2", page_icon="🔥")

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
            # Đảm bảo vector luôn trả về 512 chiều
            vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
        return vec.tolist()
    except:
        return None

# ================= UTILS =================
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

def clean_key(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT =================
def extract_pdf(file):
    specs, img_bytes = {}, None
    file.seek(0)
    pdf_content = file.read()
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    if len(doc) > 0:
        img_bytes = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
    doc.close()
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if df.empty or len(df.columns) < 2: continue
                # Tìm cột Description và cột giá trị (NEW, Sample, Spec)
                name_col, val_col = -1, -1
                for r_idx, row in df.iterrows():
                    row_str = [str(c).upper() for c in row if c]
                    if any(x in " ".join(row_str) for x in ["POM", "DESCRIPTION"]):
                        name_col = next((i for i, v in enumerate(row_str) if any(x in v for x in ["POM", "DESC"])), 0)
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

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V48.2")

with st.sidebar:
    st.header("⚙️ SETTINGS")
    # Nút dọn dẹp Database lỗi
    if st.button("🧹 Dọn dẹp mẫu lỗi"):
        db_all = supabase.table("ai_data").select("id, vector").execute()
        count_del = 0
        for item in db_all.data:
            if not item.get("vector") or len(item["vector"]) != 512:
                supabase.table("ai_data").delete().eq("id", item["id"]).execute()
                count_del += 1
        st.success(f"Đã xóa {count_del} mẫu không hợp lệ!")
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

# ================= KIỂM TRA =================
audit_file = st.file_uploader("Tải file cần đối soát (PDF)", type="pdf")

if audit_file:
    target = extract_pdf(audit_file)
    v_test = get_vector(target["img"])
    
    if v_test:
        v_test_np = np.array(v_test).reshape(1, -1)
        # Lấy lại Database mới nhất
        db_res = supabase.table("ai_data").select("*").execute()
        data_lib = db_res.data if db_res.data else []
        
        matches = []
        for item in data_lib:
            # Chỉ so sánh những mẫu có Vector chuẩn 512 chiều
            if item.get("vector") and len(item["vector"]) == 512:
                v_ref = np.array(item["vector"]).reshape(1, -1)
                score = float(cosine_similarity(v_test_np, v_ref))
                matches.append({"item": item, "score": score})
        
        if not matches:
            st.warning("⚠️ Không có dữ liệu mẫu hợp lệ trong kho. Hãy nhấn 'Dọn dẹp mẫu lỗi' và nạp lại Techpack mẫu.")
        else:
            best = sorted(matches, key=lambda x: x['score'], reverse=True)
            st.subheader(f"✨ Khớp nhất: {best[0]['item']['file_name']} ({best[0]['score']*100:.1f}%)")
            
            # Hiển thị bảng so sánh... (giữ nguyên phần cũ)
            df_res = pd.DataFrame([{
                "POM": k, "Kiểm": v, 
                "Gốc": {clean_key(nk):nv for nk,nv in best[0]['item']['spec_json'].items()}.get(clean_key(k), 0)
            } for k,v in target["specs"].items()])
            st.table(df_res)
