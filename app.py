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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V54", page_icon="🔍")

@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)
supabase = init_supabase()

@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

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

# ================= EXTRACT PDF (NÂNG CẤP TỐC ĐỘ) =================
def extract_pdf_v54(file):
    specs, img_bytes = {}, None
    try:
        file.seek(0)
        pdf_content = file.read()
        
        # 1. Chụp ảnh nhanh bằng PyMuPDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc[0].get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        doc.close()

        # 2. Trích xuất bảng (Chỉ quét trang đầu tiên chứa bảng để tránh treo)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            # Ưu tiên quét 3 trang đầu (thường thông số nằm ở đây)
            for page in pdf.pages[:3]:
                tables = page.extract_tables()
                if not tables: continue
                
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm cột POM và Giá trị
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows(): # Chỉ tìm header trong 10 dòng đầu
                        row_up = [str(c).upper().strip() for c in row if c]
                        if any(x in " ".join(row_up) for x in ["POM", "DESCRIPTION", "DIMENSION"]):
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["POM", "DESC", "DIMENSION"]): n_col = i; break
                            for i, v in enumerate(row_up):
                                if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32", "34"]): v_col = i; break
                            
                            if n_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[n_col]).strip().upper()
                                    val = parse_val(d_row[v_col])
                                    if len(name) > 3 and val > 0: specs[name] = val
                                break
        return {"specs": specs, "img": img_bytes}
    except Exception as e:
        st.error(f"Lỗi phân tích PDF: {e}")
        return None

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V54")

db_res = supabase.table("ai_data").select("*").execute()
data_lib = db_res.data if db_res.data else []

if data_lib:
    st.info(f"✅ Kho đang có **{len(data_lib)}** mẫu chuẩn. Sẵn sàng đối soát.")
else:
    st.warning("⚠️ Kho trống. Hãy nạp mẫu ở Sidebar.")

audit_file = st.file_uploader("Tải file cần đối soát (PDF)", type="pdf")

if audit_file:
    # Thêm hiệu ứng loading để biết code đang chạy
    with st.spinner("Đang trích xuất dữ liệu..."):
        target = extract_pdf_v54(audit_file)
    
    if target and target["specs"]:
        st.success(f"✨ Trích xuất thành công {len(target['specs'])} thông số.")
        
        # Tiến hành so sánh AI
        img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().cpu().numpy().reshape(1, -1).astype(np.float32)

        matches = []
        for item in data_lib:
            try:
                v_ref = np.atleast_2d(item["vector"]).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref))
                matches.append({"item": item, "score": score})
            except: continue
        
        if matches:
            best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
            st.subheader(f"🏆 Kết quả khớp nhất: {best['item']['file_name']} ({best['score']*100:.1f}%)")
            
            c1, c2 = st.columns(2)
            c1.image(target["img"], caption="Bản đang kiểm")
            c2.image(best['item']['image_url'], caption="Mẫu gốc trong kho")

            # Bảng so sánh
            ref_specs = best['item']['spec_json']
            clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): (k, v) for k, v in ref_specs.items()}
            
            diff_rows = []
            for k, v in target["specs"].items():
                k_clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                v_ref = clean_ref.get(k_clean, (None, 0))[1]
                diff = round(v - v_ref, 3)
                diff_rows.append({
                    "Hạng mục": k, "Đang kiểm": v, "Gốc": v_ref, 
                    "Chênh lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.125 else "❌ FAIL"
                })
            st.table(pd.DataFrame(diff_rows))
    else:
        st.error("❌ Không tìm thấy bảng thông số hợp lệ trong file PDF này.")
