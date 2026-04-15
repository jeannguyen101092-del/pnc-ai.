import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH & KẾT NỐI =================
# Điền URL và KEY của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor Gold", page_icon="👖")

# ================= 2. HÀM AI & XỬ LÝ DỮ LIỆU (ĐÃ FIX NHẬN DIỆN) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    """Fix: Đọc chính xác số lẻ 0.2500 và phân số ngành may"""
    try:
        if t is None or str(t).strip() == "": return 0
        # Xử lý các ký tự đặc biệt thường gặp trong bảng thông số
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        
        # Tìm các định dạng số: thập phân (40.75), phân số (1 1/2), hoặc số nguyên
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        
        v_str = match[0]
        if ' ' in v_str: # Ví dụ: "1 1/2"
            p = v_str.split()
            return float(p[0]) + eval(p[1])
        elif '/' in v_str: # Ví dụ: "1/2"
            return eval(v_str)
        else: # Ví dụ: "0.2500"
            return float(v_str)
    except:
        return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. QUÉT PDF ĐA SIZE (CHỐNG NHẦM CỘT) =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, customer = {}, None, "UNKNOWN"
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        # ====== THÊM FILTER POM CHUẨN ======
        def is_real_pom(p):
            p = str(p).upper().strip()

            # loại rác
            blacklist = [
                "DESCRIPTION","STYLE","COLOR","VENDOR",
                "FABRIC","TRIM","BUTTON","ZIPPER",
                "DRAWCORD","LABEL","WASH","FINISH",
                "CONTENT","SEASON","REFERENCE"
            ]
            if any(b in p for b in blacklist):
                return False

            # chỉ giữ measurement thật
            valid = [
                "WAIST","HIP","THIGH","INSEAM","OUTSEAM",
                "LEG","OPENING","RISE",
                "CHEST","BUST","SLEEVE","SHOULDER",
                "LENGTH"
            ]

            return any(v in p for v in valid)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    
                    n_col, size_cols = -1, {}
                    
                    # 1. TÌM HEADER
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        
                        # ===== FIX POM COLUMN =====
                        if n_col == -1:
                            for i, v in enumerate(row):
                                if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION", "POINT OF"]):
                                    sample_val = str(df.iloc[min(r_idx+1, len(df)-1), i])

                                    # 🔥 FIX: phải là text + có keyword ngành may
                                    if (
                                        not sample_val.replace('.','').isdigit() and
                                        any(k in sample_val.upper() for k in [
                                            "WAIST","HIP","THIGH","INSEAM",
                                            "CHEST","SLEEVE","LENGTH"
                                        ])
                                    ):
                                        n_col = i
                                        break
                        
                        # SIZE COLUMN (GIỮ NGUYÊN)
                        for i, v in enumerate(row):
                            if i == n_col or not v: continue
                            if any(x in v for x in ["TOL", "+/-", "NO.", "STT", "SIZE CODE"]): continue
                            if v.isdigit() or any(s == v for s in ["S","M","L","XL","2XL","3XL"]):
                                size_cols[i] = v
                        
                        if n_col != -1 and size_cols: break
                    
                    # 2. EXTRACT DATA
                    if n_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom_raw = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip()

                                # ===== FIX POM CHUẨN =====
                                if len(pom_raw) > 2 and is_real_pom(pom_raw):
                                    val = parse_val(df.iloc[d_idx, s_col])
                                    if val > 0:
                                        all_specs[s_name][pom_raw.upper()] = val

        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except:
        return None

# ================= 4. SIDEBAR & LUỒNG CHÍNH (GIỮ NGUYÊN) =================
# ... [Phần Sidebar và Upload giữ nguyên như code của bạn] ...
with st.sidebar:
    st.header("🏢 KHO MẪU")
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True)
    if new_files and st.button("NẠP KHO"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": data['customer']
                }).execute()
        st.success("Xong!"); st.rerun()

st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    target = extract_pdf_multi_size(file_audit)
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            best = df_db.sort_values('sim', ascending=False).iloc[0]
            
            st.image([target['img'], best['image_url']], width=300)
            
            # Đối soát chi tiết (Sửa lỗi khớp tên POM)
            audit_specs = target['all_specs']
            db_specs = best['spec_json']
            sel_size = st.selectbox("Chọn Size:", list(audit_specs.keys()))
            
            if sel_size in audit_specs:
                spec_audit = audit_specs[sel_size]
                # Lấy size tương ứng trong DB (ưu tiên khớp chính xác)
                spec_ref = db_specs.get(sel_size, {})
                
                rows = []
                for pom, v_audit in spec_audit.items():
                    # Fix: Tìm giá trị gốc bằng cách khớp tên POM chuẩn hóa
                    v_ref = spec_ref.get(pom, 0)
                    diff = round(v_audit - v_ref, 4)
                    status = "✅ OK" if abs(diff) < 0.001 else f"❌ LỆCH ({diff})"
                    rows.append({"POM": pom, "Audit": v_audit, "Gốc": v_ref, "Lệch": diff, "Kết quả": status})
                
                st.table(pd.DataFrame(rows))
