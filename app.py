import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH & KẾT NỐI =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96 PRO", page_icon="🏢")

# ================= 2. HÀM AI & PHÂN LOẠI THÔNG MINH =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def classify_product(specs):
    """Nhận diện cấu trúc sản phẩm dựa trên tên POM"""
    all_poms = " ".join(specs.keys()).upper()
    if any(x in all_poms for x in ["WAIST", "INSEAM", "CROTCH", "HIP"]): return "QUẦN (BOTTOM)"
    if any(x in all_poms for x in ["LAPEL", "CHEST", "BUST", "SHOULDER"]): return "ÁO/VEST/YẾM (TOP)"
    if any(x in all_poms for x in ["SWEEP", "WAIST", "SKIRT"]): return "VÁY (DRESS)"
    return "MÃ MAY MẶC"

def calculate_spec_similarity(target_spec, db_spec):
    """Tính % tương đồng thông số (Cấu trúc + Giá trị)"""
    if not target_spec or not db_spec: return 0
    t_set, d_set = set(target_spec.keys()), set(db_spec.keys())
    common = t_set.intersection(d_set)
    if not common: return 0
    
    # Sim cấu trúc (Các POM giống tên nhau)
    struct_score = len(common) / max(len(t_set), len(d_set))
    # Sim giá trị (Độ lệch số đo)
    diffs = [abs(target_spec[p] - db_spec[p]) / max(target_spec[p], db_spec[p], 1) for p in common]
    val_score = 1 - (np.mean(diffs) if diffs else 1)
    return (struct_score * 0.4 + val_score * 0.6)

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v: p = v.split(); return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().tolist()

# ================= 3. QUÉT PDF ĐA SIZE =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, customer = {}, None, "UNKNOWN"
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        full_text = " ".join([p.get_text() for p in doc])
        cust_match = re.search(r"(?i)(CUSTOMER|BUYER|CLIENT)[:\s]+([^\n]+)", full_text)
        if cust_match: customer = cust_match.group(2).strip().upper()
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 3: continue
                    n_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        if n_col == -1:
                            for i, v in enumerate(row):
                                if any(x in v for x in ["DESCRIPTION", "POM", "POSITION"]): n_col = i; break
                        for i, v in enumerate(row):
                            if i == n_col or not v or any(x in v for x in ["TOL", "+/-", "NO."]): continue
                            if v.isdigit() or any(s == v for s in ["S", "M", "L", "XL", "2XL"]): size_cols[i] = v
                        if n_col != -1 and size_cols: break
                    if n_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, n_col]).replace('\n',' ').strip().upper()
                                val = parse_val(df.iloc[d_idx, s_col])
                                if len(pom) > 3 and val > 0: all_specs[s_name][pom] = val
        return {"all_specs": all_specs, "img": img_bytes, "customer": customer}
    except: return None

# ================= 4. GIAO DIỆN & SIDEBAR =================
with st.sidebar:
    st.header("🏢 KHO MẪU")
    new_files = st.file_uploader("Nạp Techpack mới", accept_multiple_files=True)
    if new_files and st.button("XÁC NHẬN NẠP"):
        for f in new_files:
            d = extract_pdf_multi_size(f)
            if d:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_vector(d['img']),
                    "spec_json": d['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": d['customer']
                }).execute()
        st.success("Đã nạp kho!"); st.rerun()

# ================= 5. LUỒNG SO SÁNH TOP 3 & XUẤT EXCEL =================
st.title("🔍 AI SMART AUDITOR - TOP 3 SIMILARITY")
file_audit = st.file_uploader("📤 Upload file PDF Audit", type="pdf")

if file_audit:
    with st.spinner("Đang soi cấu trúc sản phẩm..."):
        target = extract_pdf_multi_size(file_audit)
    
    if target and target["all_specs"]:
        first_s = list(target["all_specs"].keys())[0]
        p_type = classify_product(target["all_specs"][first_s])
        st.info(f"📍 Nhận diện: **{p_type}** | Khách hàng: **{target['customer']}**")

        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_vector(target['img'])).reshape(1, -1)
            
            scores = []
            for i, row in df_db.iterrows():
                # Tính % Ảnh
                img_sim = float(cosine_similarity(t_vec, np.array(row['vector']).reshape(1, -1))[0][0])
                # Tính % Thông số
                db_specs = row['spec_json']
                db_first_s = list(db_specs.keys())[0] if db_specs else None
                spec_sim = calculate_spec_similarity(target['all_specs'][first_s], db_specs.get(db_first_s, {}))
                
                scores.append({"data": row, "img_sim": img_sim, "spec_sim": spec_sim, "total": (img_sim + spec_sim)/2})
            
            top_3 = sorted(scores, key=lambda x: x['total'], reverse=True)[:3]
            
            cols = st.columns(3)
            for idx, item in enumerate(top_3):
                with cols[idx]:
                    st.image(item['data']['image_url'], use_container_width=True)
                    st.write(f"**{item['data']['file_name']}**")
                    st.write(f"🖼️ Ảnh: **{item['img_sim']*100:.1f}%**")
                    st.write(f"📊 Thông số: **{item['spec_sim']*100:.1f}%**")
                    if st.button(f"Soi mẫu {idx+1}", key=f"sel_{idx}"):
                        st.session_state.selected_db = item['data']

            if 'selected_db' in st.session_state:
                st.divider()
                st.subheader(f"📊 Đối soát chi tiết: {st.session_state.selected_db['file_name']}")
                sel_size = st.selectbox("Chọn Size:", list(target['all_specs'].keys()))
                
                audit_s = target['all_specs'][sel_size]
                ref_s = st.session_state.selected_db['spec_json'].get(sel_size, {})
                
                report = []
                for pom, v_audit in audit_s.items():
                    v_ref = ref_s.get(pom, 0)
                    diff = round(v_audit - v_ref, 3)
                    report.append({"POM": pom, "Audit": v_audit, "Gốc": v_ref, "Lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.126 else "❌ LỆCH"})
                
                df_report = pd.DataFrame(report)
                st.table(df_report)
                
                # --- NÚT XUẤT EXCEL ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_report.to_excel(writer, index=False, sheet_name='Audit')
                st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), f"Report_{sel_size}.xlsx")
