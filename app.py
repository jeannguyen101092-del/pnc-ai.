import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH & STYLE =================
st.set_page_config(layout="wide", page_title="AI V20.0 - Fashion Auditor", page_icon="🛡️")

st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .css-1offfwp { background-color: #f0f2f6 !important; }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# Kết nối Supabase (Thay bằng thông tin của bạn)
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

# ================= 2. MODEL & UTILS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

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

# ================= 3. SIDEBAR (BỔ SUNG NÚT NẠP KHO) =================
with st.sidebar:
    st.title("🛡️ AI V20.0")
    
    # Hiển thị số lượng mẫu hiện có
    db_res = supabase.table("ai_data").select("id", count="exact").execute()
    st.button(f"📁 Kho mẫu: {db_res.count if db_res.count else 0} file")
    
    st.divider()
    st.subheader("🚀 NẠP FILE VÀO KHO")
    new_files = st.file_uploader("Chọn PDF mẫu để nạp", type="pdf", accept_multiple_files=True, key="upload_kho")
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            data = extract_pdf(f)
            if data and data['specs']:
                # Tạo vector
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().tolist()
                
                # Upload
                path = f"lib_{f.name.replace(' ', '_')}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url}).execute()
        st.success("Đã nạp kho thành công!")
        st.rerun()

    st.divider()
    st.write("**Size cần so sánh**")
    size = st.selectbox("Size", ["XS", "S", "M", "L", "XL"], index=2)
    st.write("**CHẾ ĐỘ TÌM KIẾM**")
    mode = st.selectbox("Mode", ["Tự động tìm", "Chọn thủ công"])

# ================= 4. MAIN (HIỂN THỊ SO SÁNH) =================
st.subheader("📊 PRODUCT SUMMARY COMPARISON")
file_audit = st.file_uploader("Upload file đối soát", type="pdf", label_visibility="collapsed", key="audit_main")

if file_audit:
    with st.spinner("Đang trích xuất và đối soát..."):
        target = extract_pdf(file_audit)
        
    if target and target["specs"]:
        db_all = supabase.table("ai_data").select("*").execute()
        if db_all.data:
            # Logic AI Matching
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1)
            
            matches = []
            for item in db_all.data:
                v_ref = np.atleast_2d(item["vector"]).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref))
                matches.append({"item": item, "score": score})
            
            best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]

            # HIỂN THỊ SONG SONG
            c1, c2 = st.columns(2)
            with c1:
                st.info("📄 BẢN ĐANG KIỂM")
                st.image(target["img"], use_container_width=True)
                st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))

            with c2:
                st.success(f"✨ MẪU GỐC (Khớp {best['score']*100:.1f}%)")
                st.image(best['item']['image_url'], use_container_width=True)
                
                ref_specs = best['item']['spec_json']
                rows = []
                for k, v in target["specs"].items():
                    k_c = re.sub(r'[^A-Z0-9]', '', k)
                    v_ref = 0
                    for rk, rv in ref_specs.items():
                        if re.sub(r'[^A-Z0-9]', '', rk) == k_c: v_ref = rv; break
                    diff = round(v - v_ref, 3)
                    rows.append({"Thông số": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": "Khớp" if abs(diff) < 0.125 else "Lệch"})
                
                df_res = pd.DataFrame(rows)
                st.table(df_res.style.applymap(lambda x: 'color: green' if x == 'Khớp' else 'color: red', subset=['Kết quả']))

            # NÚT XUẤT EXCEL
            st.divider()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False)
            st.download_button("📥 TẢI BÁO CÁO EXCEL", output.getvalue(), "Audit_Report.xlsx", type="primary")
    else:
        st.error("Không trích xuất được bảng thông số từ PDF này.")
