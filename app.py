import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Thay URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.3", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt == 'nan': return 0
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def detect_brand(text):
    text = text.upper()
    if "REITMANS" in text: return "REITMANS"
    return "OTHER"

# --- TRÍCH XUẤT PHÂN LUỒNG ---
def extract_data(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        all_text = ""
        for page in doc: all_text += page.get_text().upper() + " "
        brand = detect_brand(all_text)
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_p = (page.extract_text() or "").upper()
                
                # [LUỒNG REITMANS]
                if brand == "REITMANS":
                    if "GRADING NOT APPROVED" not in txt_p: continue
                    tables = page.extract_tables()
                    for tb in tables:
                        df = pd.DataFrame(tb)
                        p_idx, n_idx = -1, -1
                        for r_idx, row in df.iterrows():
                            row_up = [str(c).upper().strip() for c in row if c]
                            if "POM NAME" in row_up and "NEW" in row_up:
                                p_idx = row_up.index("POM NAME")
                                n_idx = row_up.index("NEW")
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[p_idx]).replace('\n',' ').strip().upper()
                                    val = parse_val(d_row[n_idx])
                                    if len(name) > 3 and val > 0: full_specs[name] = val
                                break

                # [LUỒNG CÁC HÃNG KHÁC]
                else:
                    tables = page.extract_tables()
                    for tb in tables:
                        df = pd.DataFrame(tb)
                        d_idx, v_idx = -1, -1
                        for r_idx, row in df.iterrows():
                            row_up = [str(c).upper().strip() for c in row if c]
                            if any(k in " ".join(row_up) for k in ["DESCRIPTION", "DESC", "POM NAME"]):
                                for i, v in enumerate(row):
                                    v_s = str(v).upper().strip()
                                    if any(x in v_s for x in ["DESC", "DESCRIPTION"]): d_idx = i
                                    if any(x in v_s for x in ["NEW", "ACTUAL", "REQ", "M"]): v_idx = i
                                
                                if d_idx != -1:
                                    target_col = v_idx if v_idx != -1 else d_idx + 1
                                    for d_idx_row in range(r_idx + 1, len(df)):
                                        d_row = df.iloc[d_idx_row]
                                        name = str(d_row[d_idx]).replace('\n',' ').strip().upper()
                                        val = parse_val(d_row[target_col])
                                        if len(name) > 3 and val > 0: full_specs[name] = val
                                break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu trong kho", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            # Kiểm tra trùng tên
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data:
                st.warning(f"Bỏ qua: {f.name} (Đã có)")
                continue

            d = extract_data(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "upsert":"true"})
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }).execute()
            p_bar.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V43.3")
t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_data(t_file)
    if target and target['specs']:
        st.success(f"🏷️ Thương hiệu: **{target['brand']}** | Tìm thấy **{len(target['specs'])}** hạng mục.")
        
        # Ưu tiên Brand
        db_res = supabase.table("ai_data").select("*").eq("category", target['brand']).execute()
        if not db_res.data: db_res = supabase.table("ai_data").select("*").execute()

        if db_res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim_val = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": i, "sim": sim_val})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for best in top:
                st.subheader(f"✨ Khớp nhất: {best['data']['file_name']} ({best['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Mẫu đang kiểm")
                with c2: st.image(best['data']['image_url'], caption="Mẫu gốc")

                diff_list = []
                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = re.sub(r'[^A-Z0-9]', '', p_name.upper())
                    for k_ref, val_ref in best['data']['spec_json'].items():
                        if p_clean == re.sub(r'[^A-Z0-9]', '', k_ref.upper()):
                            v_ref = val_ref
                            break
                    if v_target > 0 or v_ref > 0:
                        diff_list.append({
                            "Hạng mục": p_name, "Thông số kiểm": v_target, 
                            "Mẫu gốc": v_ref, "Lệch": round(v_target - v_ref, 2)
                        })
                
                if diff_list:
                    df = pd.DataFrame(diff_list)
                    st.table(df.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    out = io.BytesIO()
                    df.to_excel(out, index=False)
                    st.download_button("📥 Tải Báo Cáo Excel", out.getvalue(), "Audit_Report.xlsx")
            
            if st.button("🗑️ Xóa để quét tiếp"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không đọc được bảng thông số. Reitmans cần trang 'GRADING NOT APPROVED'.")
