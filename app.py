import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH (Thay thông tin thật của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI V20.0 - PRO V69", page_icon="🛡️")

# CSS làm đẹp giao diện
st.markdown("""
    <style>
    .stTable { font-size: 11px !important; }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= 3. HÀM TRÍCH XUẤT PDF V69 (SUPER EXTRACT) =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        # Tìm số, phân số (1 1/2), số thập phân
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
    try:
        file.seek(0)
        pdf_content = file.read()
        # 1. Chụp ảnh trang đầu tiên
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        # 2. Quét toàn bộ bảng thông số (Nâng cấp quét nhiều trang)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    n_col, v_col = -1, -1
                    # Quét sâu hơn để tìm Header (Dòng chứa POM hoặc Description)
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        row_str = " ".join(row_up)
                        
                        if any(x in row_str for x in ["POM", "DESCRIPTION", "DIMENSION", "MEASUREMENT"]):
                            # Tìm cột Tên
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESC", "POM", "NAME"]):
                                    n_col = i; break
                            # Tìm cột Giá trị (NEW, SPEC, SAMPLE hoặc số Size)
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["NEW", "SPEC", "SAMPLE", "32", "34", "36"]):
                                    v_col = i; break
                            
                            if n_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[n_col]).strip().upper()
                                    # Lọc bỏ các dòng nhiễu
                                    if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARKS"]): continue
                                    val = parse_val(d_row[v_col])
                                    if val > 0: specs[name] = val
                                break
                if specs: break # Dừng nếu đã tìm thấy bảng ở trang hiện tại
        return {"specs": specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("🛡️ AI V20.0 - PRO V69")
    res = supabase.table("ai_data").select("id", count="exact").execute()
    st.info(f"📁 Kho mẫu: {res.count if res.count else 0} file")
    
    st.divider()
    st.subheader("🚀 Nạp Techpack Mới")
    new_files = st.file_uploader("Upload PDF gốc vào kho", type="pdf", accept_multiple_files=True)
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            data = extract_pdf(f)
            if data and data['specs']:
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url}).execute()
        st.success("Nạp thành công!")
        st.rerun()

    if st.button("🗑️ Dọn dẹp kho"):
        supabase.table("ai_data").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        st.rerun()

# ================= 5. MAIN =================
st.subheader("📊 PRODUCT SUMMARY COMPARISON")
file_audit = st.file_uploader("Kéo thả file cần kiểm tra vào đây", type="pdf", label_visibility="collapsed")

if file_audit:
    with st.spinner("Đang tìm bảng thông số..."):
        target = extract_pdf(file_audit)
    
    if target and target["specs"]:
        st.success(f"✅ Đã trích xuất {len(target['specs'])} hạng mục thông số.")
        db_all = supabase.table("ai_data").select("*").execute()
        
        if db_all.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
            
            matches = []
            for item in db_all.data:
                if item.get("vector") and len(item["vector"]) == 512:
                    v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                    score = float(cosine_similarity(v_test, v_ref))
                    matches.append({"item": item, "score": score})
            
            if matches:
                best_match = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                best = best_match['item']
                score_pct = best_match['score'] * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.info("📄 BẢN ĐANG KIỂM TRA")
                    st.image(target["img"], use_container_width=True)
                    st.table(pd.DataFrame([{"POM": k, "Số đo": v} for k,v in target["specs"].items()]))
                
                with col2:
                    st.success(f"✨ MẪU GỐC TRONG KHO (Khớp {score_pct:.1f}%)")
                    st.image(best['image_url'], use_container_width=True)
                    
                    ref_specs = best['spec_json']
                    rows = []
                    clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): v for k, v in ref_specs.items()}
                    
                    for k, v in target["specs"].items():
                        k_c = re.sub(r'[^A-Z0-9]', '', k.upper())
                        v_ref = clean_ref.get(k_c, 0)
                        diff = round(v - v_ref, 3)
                        rows.append({"Thông số": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": "Khớp" if abs(diff) < 0.125 else "Lệch"})
                    
                    df_res = pd.DataFrame(rows)
                    st.table(df_res.style.applymap(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))
    else:
        st.error("❌ Không trích xuất được bảng thông số. PDF này có thể là file Scan ảnh hoặc bảng nằm ở trang quá xa.")
