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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V94", page_icon="📏")

# ================= 2. MODEL AI & UTILS =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    if any(x in t for x in ["PANT", "JEAN", "SHORT", "QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "TEE", "JACKET", "ÁO"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "VÁY"]): return "VÁY/ĐẦM"
    return "KHÁC"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or any(x in txt for x in ["mm", "yd", "gr", "kg", "pcs"]): return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
    return vec.astype(float).tolist()

# ================= 3. TRÍCH XUẤT THÔNG MINH =================
def extract_pdf_v94(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc: full_text += page.get_text()
        doc.close()
        
        category = detect_category(full_text, file.name)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST", "CHEST", "HIP", "LENGTH", "SHOULDER"] if k in flat_text) < 2: continue

                    # Dò cột thông minh
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "POM NAME", "POSITION"]): n_col = i; break
                    
                    if n_col != -1:
                        max_nums = 0
                        for i in range(len(df.columns)):
                            if i == n_col: continue
                            num_count = sum(1 for val in df.iloc[:12, i] if parse_val(val) > 0)
                            if num_count > max_nums: max_nums = num_count; v_col = i

                    if n_col != -1 and v_col != -1:
                        for d_idx in range(len(df)):
                            name = str(df.iloc[d_idx, n_col]).replace('\n', ' ').strip().upper()
                            val = parse_val(df.iloc[d_idx, v_col])
                            if len(name) > 3 and val > 0 and "POM" not in name: specs[name] = val
                if specs: break 
        return {"specs": specs, "img": img_bytes, "category": category}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO THIẾT KẾ")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng số mẫu", f"{res_db.count or 0} file")
    
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True)
    if new_files and st.button("🚀 XÁC NHẬN NẠP"):
        for f in new_files:
            data = extract_pdf_v94(f)
            if data and data['specs']:
                vec = get_image_vector(data['img'])
                path = f"lib_{f.name.replace(' ','_')}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category']}).execute()
        st.rerun()

# ================= 5. MAIN FLOW =================
st.title("🔍 AI SMART AUDITOR V94")
file_audit = st.file_uploader("📤 Upload file cần kiểm tra", type="pdf")

if file_audit:
    target = extract_pdf_v94(file_audit)
    if target and target["specs"]:
        st.success(f"Phân loại: {target['category']} | {len(target['specs'])} vị trí đo.")
        
        res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        if res.data:
            # So sánh tìm TOP 3
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            all_matches = []
            for item in res.data:
                sim = cosine_similarity(target_vec, np.array(item['vector']).reshape(1, -1))[0][0]
                all_matches.append({**item, "sim": sim})
            
            # Lấy 3 mã cao nhất
            top_3 = sorted(all_matches, key=lambda x: x['sim'], reverse=True)[:3]
            
            # Hiển thị Top 3 mẫu tương đồng
            st.subheader("🖼️ TOP 3 MẪU TƯƠNG ĐỒNG NHẤT")
            cols = st.columns(3)
            for i, match in enumerate(top_3):
                with cols[i]:
                    st.image(match['image_url'], caption=f"Top {i+1}: {match['file_name']} ({match['sim']:.1%})")

            # Chọn 1 mẫu để so sánh chi tiết bảng (Mặc định là mẫu số 1)
            selected_master = st.selectbox("Chọn mẫu gốc để đối soát chi tiết:", [m['file_name'] for m in top_3])
            master_data = next(m for m in top_3 if m['file_name'] == selected_master)

            # Bảng đối soát
            st.subheader(f"📊 Bảng so sánh với: {selected_master}")
            audit_list = []
            for pom, val in target['specs'].items():
                m_val = master_data['spec_json'].get(pom, 0)
                diff = round(val - m_val, 3) if m_val else 0
                status = "✅ Khớp" if abs(diff) < 0.126 else f"❌ Lệch ({diff})"
                audit_list.append({"POM": pom, "File Mới": val, "Mẫu Gốc": m_val, "Kết quả": status})
            
            df_audit = pd.DataFrame(audit_list)
            st.table(df_audit)

            # --- XUẤT EXCEL ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_audit.to_excel(writer, index=False, sheet_name='Audit_Report')
            st.download_button(label="📥 Tải Báo Cáo Excel", data=output.getvalue(), file_name=f"Audit_{selected_master}.xlsx")
        else:
            st.warning("Chưa có mẫu cùng chủng loại trong kho.")

st.caption("AI Fashion Auditor V94 - Support Top 3 Similarity & Excel Export")
