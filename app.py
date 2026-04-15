import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from rapidfuzz import process, fuzz

# ================= 1. CẤU HÌNH & KẾT NỐI =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96 Gold", page_icon="👖")

# ================= 2. HÀM AI & ĐỌC SỐ ĐO CHÍNH XÁC =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    """Đọc chính xác các số thập phân dài như 0.2500 hoặc 40.7500"""
    try:
        if t is None or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        # Tìm tất cả định dạng số (thập phân, phân số, nguyên)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split(); return float(p[0]) + eval(p[1])
        return eval(v_str) if '/' in v_str else float(v_str)
    except: return 0

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().tolist()

# ================= 3. QUÉT PDF: ƯU TIÊN LẤY TÊN VỊ TRÍ (CHỮ) =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, customer = {}, None, "UNKNOWN"
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 3: continue
                    
                    n_col, size_cols = -1, {}
                    # Bước 1: Tìm cột chứa TÊN VỊ TRÍ (Description/POM Name)
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        if n_col == -1:
                            for i, v in enumerate(row):
                                # Loại bỏ "MEASUREMENT" vì cột đó thường chứa số đo
                                if any(x in v for x in ["DESCRIPTION", "POSITION", "POM NAME", "POINT OF"]):
                                    # Kiểm tra dòng tiếp theo xem có phải là chữ (Leg Opening, Hip...) không
                                    test_val = str(df.iloc[min(r_idx+1, len(df)-1), i])
                                    if not test_val.replace('.','').isdigit():
                                        n_col = i; break
                        
                        # Bước 2: Tìm cột chứa SIZE
                        for i, v in enumerate(row):
                            if i == n_col or not v: continue
                            if any(x in v for x in ["TOL", "+/-", "NO.", "STT"]): continue
                            if v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "2XL", "3XL"]):
                                size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    
                    # Bước 3: Đổ dữ liệu (Chỉ lấy tên nếu là CHỮ)
                    if n_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom_raw = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip()
                                # Chỉ lấy nếu tên vị trí chứa ít nhất 1 ký tự chữ
                                if len(pom_raw) > 2 and re.search('[a-zA-Z]', pom_raw):
                                    val = parse_val(df.iloc[d_idx, s_col])
                                    if val > 0: all_specs[s_name][pom_raw.upper()] = val
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except: return None

# ================= 4. GIAO DIỆN ĐỐI SOÁT & XUẤT EXCEL =================
st.title("🔍 AI SMART AUDITOR - V96 GOLD")
file_audit = st.file_uploader("📤 Upload file PDF Audit", type="pdf")

if file_audit:
    with st.spinner("AI đang soi chi tiết từng vị trí..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            best = df_db.sort_values('sim', ascending=False).iloc[0]
            
            st.divider()
            audit_all, db_all = target['all_specs'], best['spec_json']
            sel_size = st.selectbox("Chọn Size đối soát:", list(audit_all.keys()))
            
            if sel_size in audit_all:
                spec_audit = audit_all[sel_size]
                # So khớp Size mờ để lấy từ DB
                db_sz_res = process.extractOne(sel_size, db_all.keys(), scorer=fuzz.Ratio)
                spec_ref = db_all.get(db_sz_res[0], {}) if db_sz_res else {}
                
                report = []
                for pom_audit, v_audit in spec_audit.items():
                    # So khớp Tên Vị Trí mờ (Ví dụ: "Waist Band" khớp với "Waistband")
                    match_res = process.extractOne(pom_audit, spec_ref.keys(), scorer=fuzz.TokenSortRatio)
                    v_ref = spec_ref.get(match_res[0], 0) if match_res and match_res[1] > 80 else 0
                    
                    diff = round(v_audit - v_ref, 4)
                    status = "✅ OK" if abs(diff) < 0.0001 else f"❌ LỆCH ({diff:+.4f})"
                    
                    report.append({"Vị trí (POM)": pom_audit, "Audit": v_audit, "Gốc": v_ref, "Chênh lệch": diff, "Kết quả": status})
                
                df_final = pd.DataFrame(report)
                st.table(df_final) # Hiển thị bảng sạch sẽ như yêu cầu
                
                # Xuất Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_final.to_excel(writer, index=False, sheet_name='Audit')
                st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Audit_{sel_size}.xlsx")
