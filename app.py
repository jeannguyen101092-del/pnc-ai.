import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH (THAY THÔNG TIN THẬT) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI V20.0 - PRO", page_icon="🛡️")

# CSS Style
st.markdown("""
    <style>
    .stTable { font-size: 12px !important; }
    .css-1offfwp { background-color: #f0f2f6 !important; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #1f77b4; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= 3. HÀM XỬ LÝ DỮ LIỆU =================
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
        # Lấy ảnh preview trang 1
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        # Trích xuất bảng
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages[:2]: # Chỉ quét 2 trang đầu để tránh treo
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
    st.title("🛡️ AI V20.0 - PRO")
    res = supabase.table("ai_data").select("id", count="exact").execute()
    st.info(f"📁 Kho mẫu: {res.count if res.count else 0} file")
    
    st.divider()
    st.subheader("🚀 NẠP MẪU MỚI")
    new_files = st.file_uploader("Upload Techpack gốc", type="pdf", accept_multiple_files=True)
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            data = extract_pdf(f)
            if data and data['specs'] and data['img']:
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().tolist()
                
                # FIX TÊN FILE: Đảm bảo đuôi .png chuẩn
                safe_name = f.name.replace(".pdf", "").replace(" ", "_")
                path = f"lib_{safe_name}.png"
                
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url}).execute()
        st.success("Đã nạp kho thành công!")
        st.rerun()

    if st.button("🗑️ Dọn dẹp kho (Xóa hết)"):
        supabase.table("ai_data").delete().neq("id", 0).execute()
        st.rerun()

# ================= 5. MAIN =================
st.subheader("📊 PRODUCT SUMMARY COMPARISON")
file_audit = st.file_uploader("Kéo thả file cần kiểm tra vào đây", type="pdf", label_visibility="collapsed")

if file_audit:
    target = extract_pdf(file_audit)
    if target and target["specs"]:
        db_all = supabase.table("ai_data").select("*").execute()
        if db_all.data:
            # AI Matching
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
                best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("📄 BẢN ĐANG KIỂM")
                    st.image(target["img"], use_container_width=True)
                    st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))
                
                with c2:
                    st.success(f"✨ MẪU GỐC (Khớp {best['score']*100:.1f}%)")
                    # Hiển thị ảnh mẫu từ URL công khai
                    st.image(best['item']['image_url'], use_container_width=True)
                    
                    # So sánh bảng
                    ref_specs = best['item']['spec_json']
                    rows = []
                    for k, v in target["specs"].items():
                        k_c = re.sub(r'[^A-Z0-9]', '', k.upper())
                        v_ref = 0
                        for rk, rv in ref_specs.items():
                            if re.sub(r'[^A-Z0-9]', '', rk.upper()) == k_c: v_ref = rv; break
                        diff = round(v - v_ref, 3)
                        rows.append({"Thông số": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": "Khớp" if abs(diff) < 0.125 else "Lệch"})
                    
                    df_res = pd.DataFrame(rows)
                    st.table(df_res.style.applymap(lambda x: 'color: green' if x == 'Khớp' else 'color: red', subset=['Kết quả']))
            else:
                st.warning("⚠️ Kho chưa có dữ liệu vector chuẩn. Hãy dọn dẹp và nạp lại.")
