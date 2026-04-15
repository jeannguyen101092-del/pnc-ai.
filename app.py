import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor", page_icon="👖")

# ================= 2. HÀM AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    try:
        if t is None or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v_str)) if '/' in v_str else float(v_str)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. FIX PDF =================
def extract_pdf_multi_size(file):
    all_specs, img_bytes, customer = {}, None, "UNKNOWN"
    try:
        file.seek(0); pdf_content = file.read()

        txt_all = ""
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for p in pdf.pages[:2]:
                txt_all += p.extract_text() or ""
        if "REIMANT" in txt_all.upper():
            customer = "REIMANT"

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2:
                        continue

                    desc_col = -1
                    size_cols = {}

                    for r_idx in range(min(10, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]

                        for i, v in enumerate(row):
                            if customer == "REIMANT":
                                if "POM NAME" in v:
                                    desc_col = i
                            else:
                                if "DESCRIPTION" in v or "POM" in v:
                                    desc_col = i

                        for i, v in enumerate(row):
                            if i == desc_col or not v:
                                continue
                            if any(x in v for x in ["TOL", "+/-", "CODE", "DIM"]):
                                continue
                            if v.isdigit() or v in ["XS","S","M","L","XL","2XL","3XL"]:
                                size_cols[i] = v

                    if desc_col != -1 and size_cols:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs:
                                all_specs[s_name] = {}

                            for d_idx in range(len(df)):
                                raw_desc = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()

                                desc = re.sub(r'^\d+[\.\-\)]*\s*', '', raw_desc)
                                desc = re.sub(r'\s+', ' ', desc).strip()

                                if len(desc) < 3:
                                    continue
                                if desc.upper() in ["DESCRIPTION", "POM NAME"]:
                                    continue

                                val = parse_val(df.iloc[d_idx, s_col])
                                if val > 0:
                                    all_specs[s_name][desc] = val

        return {"all_specs": all_specs, "img": img_bytes}

    except Exception as e:
        print("ERROR extract:", e)
        return None


# ================= 4. UI =================
with st.sidebar:
    st.header("🏢 QUẢN LÝ KHO")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count if res_count.count else 0
    st.metric("Số lượng mẫu trong kho", f"{count} mẫu")
    
    new_files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True)
    if new_files and st.button("NẠP KHO"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.success("Đã nạp!"); st.rerun()

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
            
            c1, c2 = st.columns(2)
            c1.image(target['img'], caption="File Hiện Tại", width=300)
            c2.image(best['image_url'], caption=f"Mẫu Khớp Nhất (Sim: {best['sim']:.1%})", width=300)
            
            sel_size = st.selectbox("Chọn Size:", list(target['all_specs'].keys()))
            spec_audit = target['all_specs'][sel_size]
            spec_ref = best['spec_json'].get(sel_size, list(best['spec_json'].values())[0])
            
            report = []

            # ===== normalize =====
            def norm_key(x):
                x = str(x).lower().strip()
                x = re.sub(r'\(.*?\)', '', x)
                x = re.sub(r'[^a-z0-9 ]', '', x)
                x = re.sub(r'\s+', ' ', x)
                return x

            ref_map = {norm_key(k): v for k, v in spec_ref.items()}

            for pom, val in spec_audit.items():
                k_norm = norm_key(pom)

                ref_val = ref_map.get(k_norm, 0)

                if ref_val == 0:
                    for k, v in ref_map.items():
                        if k_norm in k or k in k_norm:
                            ref_val = v
                            break

                diff = round(val - ref_val, 3)

                report.append({
                    "Thông số": pom,
                    "Thực tế": val,
                    "Mẫu kho": ref_val,
                    "Lệch": diff,
                    "Kết quả": "✅ OK" if abs(diff) < 0.2 else "❌ Lệch"
                })
            
            df_rep = pd.DataFrame(report)
            st.table(df_rep)
            
            towrite = io.BytesIO()
            df_rep.to_excel(towrite, index=False, engine='xlsxwriter')
            st.download_button("📥 Xuất báo cáo Excel", data=towrite.getvalue(), file_name="report_audit.xlsx")
    else:
        st.warning("⚠️ Không tìm thấy bảng thông số.")
