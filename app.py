import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.5", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- HÀM XỬ LÝ PHÂN SỐ (16 1/2 -> 16.5) ---
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt == 'nan' or txt == '-': return 0
        # Tìm các định dạng: "16 1/2", "16.5", "16"
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v: # Trường hợp "16 1/2"
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- HÀM LÀM SẠCH TÊN VỊ TRÍ ĐỂ KHỚP ---
def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT ĐÚNG CỘT KHOANH TRÒN ---
def extract_reitmans_pom_new(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        all_text = "".join([p.get_text() for p in doc]).upper()
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_p = (page.extract_text() or "").upper()
                # Chỉ lấy trang Grading của Reitmans
                if brand == "REITMANS" and "GRADING NOT APPROVED" not in txt_p: continue
                
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    p_name_idx, new_idx = -1, -1
                    
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        # Tìm đúng tiêu đề cột bạn khoanh tròn
                        if "POM NAME" in row_up and "NEW" in row_up:
                            p_name_idx = row_up.index("POM NAME")
                            new_idx = row_up.index("NEW")
                            
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                # Bỏ qua dòng ghi chú (thường bắt đầu bằng REF, RELATED hoặc quá ngắn)
                                if len(name) < 4 or any(x in name for x in ["REF:", "RELATED:"]): continue
                                
                                val_new = parse_reitmans_val(d_row[new_idx])
                                if val_new > 0:
                                    full_specs[name] = val_new
                            break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except: return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Mẫu đã nạp", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp Techpack Reitmans", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):
            # Chống nạp trùng
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data:
                st.warning(f"Đã có: {f.name}")
                continue

            d = extract_reitmans_pom_new(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path), "category": d['brand']
                }).execute()
                st.success(f"Đã nạp: {f.name}")
            p.progress((i + 1) / len(files))
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V43.5")
t_file = st.file_uploader("Upload file cần so sánh thông số", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_reitmans_pom_new(t_file)
    if target and target['specs']:
        st.info(f"✅ Đã trích xuất **{len(target['specs'])}** hạng mục từ cột NEW.")
        
        # Tìm trong kho theo Brand
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
                    sim = float(cosine_similarity(v_test, v_ref)) * 100
                    matches.append({"data": i, "sim": sim})
            
            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]
            for m in top:
                st.subheader(f"✨ Khớp mẫu: {m['data']['file_name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                with c1: st.image(target['img'], caption="Bản vẽ đang kiểm")
                with c2: st.image(m['data']['image_url'], caption="Mẫu gốc")

                # --- SO SÁNH THÔNG SỐ NEW ---
                diff_list = []
                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = clean_pos(p_name)
                    # Tìm hạng mục tương ứng trong mẫu gốc
                    for k_ref, val_ref in m['data']['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break
                    
                    if v_target > 0 or v_ref > 0:
                        diff_list.append({
                            "Vị trí (POM Name)": p_name,
                            "Thông số NEW (Kiểm)": v_target,
                            "Thông số NEW (Gốc)": v_ref,
                            "Chênh lệch": round(v_target - v_ref, 2)
                        })
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    # Bôi đỏ dòng lệch > 0.5
                    st.table(df_r.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Chênh lệch']))
                    
                    out = io.BytesIO()
                    df_r.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel đối soát", out.getvalue(), "Audit_New_Only.xlsx")
            
            if st.button("🗑️ Xóa file vừa quét"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không lấy được thông số. Hãy đảm bảo PDF đúng trang GRADING NOT APPROVED và có cột NEW.")
