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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.1", page_icon="📊")

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
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- TRÍCH XUẤT SIÊU MẠNH (AUTO-DETECT BẢNG) ---
def extract_flexible_data(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        
        all_text = ""
        for page in doc: all_text += page.get_text().upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_p = (page.extract_text() or "").upper()
                
                # Nếu là hàng Reitmans, ưu tiên tìm trang Grading. 
                # Nếu không thấy Reitmans, quét mọi bảng có số đo.
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if len(df.columns) < 2: continue
                    
                    p_name_idx, val_idx = -1, -1
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        # Tìm cột tên vị trí (POM NAME hoặc DESC)
                        if any(k in " ".join(row_up) for k in ["POM NAME", "DESCRIPTION", "DESC", "POM CODE"]):
                            for i, val in enumerate(row):
                                v_s = str(val).upper().strip()
                                if any(x in v_s for x in ["POM NAME", "DESC"]): p_name_idx = i
                                if any(x in v_s for x in ["NEW", "ACTUAL", "REQ", "M"]): val_idx = i
                            
                            # Nếu tìm thấy cột tên và cột giá trị
                            if p_name_idx != -1:
                                target_val_idx = val_idx if val_idx != -1 else p_name_idx + 1
                                for d_idx in range(r_idx + 1, len(df)):
                                    try:
                                        d_row = df.iloc[d_idx]
                                        name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                        val_num = parse_val(d_row[target_val_idx])
                                        if len(name) > 3 and val_num > 0:
                                            full_specs[name] = val_num
                                    except: continue
                            break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except Exception as e:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Tổng mẫu hiện có", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp mẫu mới (PDF)", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        status = st.empty()
        for i, f in enumerate(files):
            d = extract_flexible_data(f)
            if d and d['specs'] and d['img']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "upsert":"true"})
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }, on_conflict="file_name").execute()
                status.success(f"Đã nạp: {f.name}")
            else:
                status.error(f"Lỗi: {f.name} (Không tìm thấy bảng thông số)")
            p_bar.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V43.1")
t_file = st.file_uploader("Upload file cần đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_flexible_data(t_file)
    if target and target['specs']:
        st.success(f"✅ Đã trích xuất **{len(target['specs'])}** hạng mục thông số.")
        
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": i, "sim": sim})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for best in top:
                st.subheader(f"✨ Khớp mẫu: {best['data']['file_name']} ({best['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(best['data']['image_url'], caption="Mẫu trong kho")

                diff_list = []
                for p_name, v_ kiểm in target['specs'].items():
                    v_gốc = 0
                    p_clean = re.sub(r'[^A-Z0-9]', '', p_name.upper())
                    for k_ref, val_ref in best['data']['spec_json'].items():
                        if p_clean == re.sub(r'[^A-Z0-9]', '', k_ref.upper()):
                            v_gốc = val_ref
                            break
                    
                    if v_kiểm > 0 or v_gốc > 0:
                        diff_list.append({
                            "Hạng mục": p_name, "Thông số kiểm": v_kiểm, "Mẫu gốc": v_gốc,
                            "Lệch": round(v_kiểm - v_gốc, 2)
                        })
                
                if diff_list:
                    df = pd.DataFrame(diff_list)
                    st.table(df.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Lệch']))
                    
                    out = io.BytesIO()
                    df.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), "Audit_Report.xlsx")
            
            if st.button("🗑️ Xóa file để quét tiếp"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không đọc được bảng thông số. Hãy kiểm tra PDF có bảng POM không.")
