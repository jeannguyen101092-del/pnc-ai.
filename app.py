import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI V20.0 - PRO V72", page_icon="🛡️")

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= 3. TRÍCH XUẤT (ƯU TIÊN DESCRIPTION) =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_pdf(file):
    specs, img_bytes = {}, None
    try:
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
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # ƯU TIÊN DESCRIPTION TRƯỚC
                        if any(x in " ".join(row_up) for x in ["DESCRIPTION", "POM"]):
                            # Tìm cột Description
                            for i, v in enumerate(row_up):
                                if "DESCRIPTION" in v or "DESC" in v:
                                    n_col = i; break
                            # Nếu không thấy Description mới lấy POM
                            if n_col == -1:
                                for i, v in enumerate(row_up):
                                    if "POM" in v: n_col = i; break
                                    
                            # Tìm cột giá trị
                            for i, v in enumerate(row_up):
                                if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32", "34"]):
                                    v_col = i; break
                            
                            if n_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                    if len(name) < 3 or any(x in name for x in ["TOL", "REF"]): continue
                                    val = parse_val(d_row[v_col])
                                    if val > 0: specs[name] = val
                                break
                if specs: break
        return {"specs": specs, "img": img_bytes}
    except Exception as e:
        st.error(f"Lỗi extract: {e}")
        return None

# ================= 4. SIDEBAR (FIX NẠP KHO) =================
with st.sidebar:
    st.header("🛡️ AI V20.0 - PRO V72")
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.info(f"📁 Kho mẫu: {res_count.count if res_count.count else 0} file")
    except: st.error("Lỗi kết nối Supabase")
    
    st.divider()
    st.subheader("🚀 NẠP TECHPACK MỚI")
    new_files = st.file_uploader("Upload PDF nạp kho", type="pdf", accept_multiple_files=True)
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            with st.spinner(f"Đang nạp: {f.name}..."):
                data = extract_pdf(f)
                if data and data['specs'] and data['img']:
                    # 1. Tạo Vector AI
                    img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                    
                    # 2. Upload Ảnh & Data lên Supabase
                    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))
                    path = f"lib_{safe_name}.png"
                    
                    try:
                        # Thêm content-type để ảnh hiển thị được
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        
                        supabase.table("ai_data").insert({
                            "file_name": f.name, 
                            "vector": vec, 
                            "spec_json": data['specs'], 
                            "image_url": url
                        }).execute()
                        st.toast(f"✅ Thành công: {f.name}")
                    except Exception as e:
                        st.error(f"Lỗi Supabase với file {f.name}: {e}")
                else:
                    st.warning(f"⚠️ Bỏ qua {f.name}: Không tìm thấy bảng thông số.")
        st.success("Đã hoàn tất quá trình nạp!")
        st.rerun()

    if st.button("🗑️ Dọn dẹp kho"):
        supabase.table("ai_data").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        st.rerun()

# ================= 5. MAIN (SO SÁNH) =================
# Giữ nguyên phần Main từ V71...
