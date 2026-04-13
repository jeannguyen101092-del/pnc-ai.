import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CONFIG & STYLE =================
st.set_page_config(layout="wide", page_title="AI V20.0 - Fashion Auditor", page_icon="🛡️")

st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    [data-testid="stMetricValue"] { font-size: 22px; }
    .css-1offfwp { background-color: #262730 !important; }
    </style>
    """, unsafe_allow_html=True)

# Kết nối Supabase
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)

# ================= 2. MODEL (FIX LỖI NAMEERROR) =================
@st.cache_resource
def load_model(): # Định nghĩa tên hàm là load_model để khớp với dòng gọi hàm bên dưới
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model() # Gọi hàm chính xác

# ================= 3. UTILS =================
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

def extract_pdf(file):
    specs, img_bytes = {}, None
    try:
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                for tb in page.extract_tables():
                    df = pd.DataFrame(tb)
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row if c]
                        if any(x in " ".join(row_up) for x in ["POM", "DESCRIPTION"]):
                            n_idx, v_idx = 0, 1
                            for i, v in enumerate(row_up):
                                if any(x in v for x in ["DESC", "POM"]): n_idx = i
                                if any(x in v for x in ["NEW", "SAMPLE", "M", "32"]): v_idx = i
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_idx]).upper()
                                val = parse_val(df.iloc[d_idx, v_idx])
                                if len(name) > 3 and val > 0: specs[name] = val
                            break
        return {"specs": specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.title("🛡️ AI V20.0")
    st.button("📁 Kho mẫu: 6 file", use_container_width=True)
    st.button("🧩 CẬP NHẬT KHO MẪU", use_container_width=True)
    st.divider()
    st.write("**Size cần so sánh**")
    size = st.selectbox("Size", ["XS", "S", "M", "L", "XL"], index=2, label_visibility="collapsed")
    st.write("**CHỌN MẪU THỦ CÔNG**")
    mode = st.selectbox("Chế độ", ["Tự động tìm", "Chọn từ kho"], label_visibility="collapsed")

# ================= 5. MAIN =================
st.subheader("📊 PRODUCT SUMMARY COMPARISON")
file = st.file_uploader("Upload Techpack", type="pdf", label_visibility="collapsed")

if file:
    target = extract_pdf(file)
    if target and target["specs"]:
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            # AI Matching
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1)
            
            matches = []
            for item in db_res.data:
                v_ref = np.atleast_2d(item["vector"]).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref))
                matches.append({"item": item, "score": score})
            
            best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]

            # DISPLAY SIDE-BY-SIDE
            col1, col2 = st.columns(2)
            with col1:
                st.info("📄 BẢN ĐANG KIỂM TRA")
                st.image(target["img"], use_container_width=True)
                df_l = pd.DataFrame([{"STT": i+1, "Hạng mục": k, "Số đo": v} for i, (k,v) in enumerate(target["specs"].items())])
                st.table(df_l)

            with col2:
                st.success(f"✨ MẪU GỐC (Khớp {best['score']*100:.1f}%)")
                st.image(best['item']['image_url'], use_container_width=True)
                
                # So sánh logic
                ref_specs = best['item']['spec_json']
                data_r = []
                for k, v in target["specs"].items():
                    k_clean = re.sub(r'[^A-Z0-9]', '', k)
                    v_ref = 0
                    for rk, rv in ref_specs.items():
                        if re.sub(r'[^A-Z0-9]', '', rk) == k_clean:
                            v_ref = rv; break
                    diff = round(v - v_ref, 3)
                    res = "Khớp" if abs(diff) < 0.125 else "Lệch"
                    data_r.append({"Thông số": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": res})
                
                df_r = pd.DataFrame(data_r)
                st.table(df_r.style.applymap(lambda x: 'color: green' if x == 'Khớp' else 'color: red', subset=['Kết quả']))

            # XUẤT EXCEL
            st.divider()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_r.to_excel(writer, index=False, sheet_name='Audit')
            
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), "Audit_Report.xlsx", "application/vnd.ms-excel", type="primary")
