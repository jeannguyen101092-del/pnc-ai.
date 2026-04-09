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

st.set_page_config(layout="wide", page_title="AI Fashion Pro V41.4", page_icon="📊")

if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0
if "audit_key" not in st.session_state: st.session_state.audit_key = 0

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- XỬ LÝ SỐ ĐO PHÂN SỐ (14 1/2 -> 14.5) ---
def parse_val_fraction(t):
    try:
        t = str(t).replace(',', '.').strip().lower()
        if not t or t == 'nan': return 0
        # Tìm các định dạng số: 14 1/2, 14.5, 14
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- TRÍCH XUẤT SIÊU MẠNH (REITMANS FOCUS) ---
def extract_techpack_v414(pdf_file):
    full_specs, img_bytes, all_txt = {}, None, ""
    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                txt_page = (page.extract_text() or "").upper()
                all_txt += txt_page + " "
                
                tables = page.extract_tables()
                for tb in tables:
                    if len(tb) < 3: continue # Bỏ qua bảng quá nhỏ
                    df = pd.DataFrame(tb)
                    
                    desc_idx = -1
                    size_cols = {}

                    # 1. TÌM DÒNG HEADER CHỨA "POM NAME"
                    for idx, row in df.iterrows():
                        row_str = [str(c).upper().strip() for c in row if c]
                        row_full = " ".join(row_str)
                        
                        if "POM NAME" in row_full or "DESCRIPTION" in row_full:
                            # Xác định vị trí các cột
                            for i, val in enumerate(row):
                                val_s = str(val).upper().strip()
                                # Chốt cột POM Name làm hạng mục
                                if "POM NAME" in val_s or "DESCRIPTION" in val_s: desc_idx = i
                                # Nhận diện Size (Cột là Số: 24, 26... hoặc S, M, L)
                                elif val and (val_s.isdigit() or (len(val_s) <= 3 and val_s not in ["TOL", "NO", "+", "-"])):
                                    size_cols[val_s] = i
                            
                            # 2. LẤY DỮ LIỆU CÁC DÒNG TIẾP THEO
                            if desc_idx != -1:
                                for d_idx in range(idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    pos_name = str(d_row[desc_idx]).replace('\n',' ').strip().upper()
                                    
                                    # Lọc rác: Bỏ qua dòng trống, dòng ghi chú quá dài hoặc mã code
                                    if len(pos_name) < 3 or "SEE PAGE" in pos_name: continue
                                    
                                    if pos_name not in full_specs: full_specs[pos_name] = {}
                                    for s_name, s_idx in size_cols.items():
                                        v_num = parse_val_fraction(d_row[s_idx])
                                        if v_num > 0: full_specs[pos_name][s_name] = v_num
                            break # Đã tìm thấy bảng POM ở trang này, chuyển trang
                            
        return {"specs": full_specs, "img": img_bytes, "text": all_txt}
    except: return None

# --- SIDEBAR: NẠP KHO ---
with st.sidebar:
    st.header("📂 HỆ THỐNG REITMANS")
    res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Kho đang lưu trữ", f"{res_db.count} mẫu")

    files = st.file_uploader("Nạp Techpacks mẫu", accept_multiple_files=True, key=f"f_{st.session_state.uploader_key}")
    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_techpack_v414(f)
            if d and d['specs']:
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                img_path = f"lib_{f.name.replace('.pdf', '.png')}"
                supabase.storage.from_(BUCKET).upload(path=img_path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(img_path),
                    "category": "bottoms" if any(k in d for k in ["INSEAM", "WAIST", "PANT"]) else "tops"
                }).execute()
            p_bar.progress((i + 1) / len(files))
        st.session_state.uploader_key += 1
        st.success("✅ Nạp thành công!"); st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V41.4")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf", key=f"a_{st.session_state.audit_key}")

if test_file:
    target = extract_techpack_v414(test_file)
    if target and target['specs']:
        # 1. Danh sách Size Số (24, 25, 26...)
        all_sizes = sorted(list({s for p in target['specs'].values() for s in p.keys()}), key=lambda x: int(x) if x.isdigit() else 0)
        sel_size = st.selectbox("🎯 CHỌN SIZE ĐỐI SOÁT:", all_sizes)

        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)) * 100
                    matches.append({"data": i, "sim": sim})
            
            m = sorted(matches, key=lambda x: x['sim'], reverse=True)
            if m:
                best = m
                st.subheader(f"✨ Khớp mẫu: {best['data']['file_name']} ({best['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                c1.image(target['img'], caption="Mẫu đang kiểm")
                c2.image(best['data']['image_url'], caption="Mẫu trong kho")

                # BẢNG ĐỐI SOÁT CHI TIẾT
                diff_list = []
                for p_name, p_vals in target['specs'].items():
                    v1 = p_vals.get(sel_size, 0)
                    # Khớp vị trí (Hạng mục) thông minh: Lấy 10 ký tự đầu
                    v2 = 0
                    for k_ref, v_ref_map in best['data']['spec_json'].items():
                        if p_name[:12] in k_ref: # Khớp 12 ký tự để chính xác hơn
                            v2 = v_ref_map.get(sel_size, 0)
                            break
                    
                    if v1 > 0 or v2 > 0:
                        diff_list.append({
                            "Hạng mục (POM Name)": p_name,
                            f"Thực tế ({sel_size})": v1,
                            f"Gốc ({sel_size})": v2,
                            "Chênh lệch": round(v1 - v2, 2)
                        })
                
                if diff_list:
                    df_res = pd.DataFrame(diff_list)
                    st.table(df_res.style.map(lambda x: 'color: red; font-weight: bold' if abs(x) > 0.5 else 'color: white', subset=['Chênh lệch']))
                    
                    out = io.BytesIO()
                    df_res.to_excel(out, index=False)
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), f"Audit_{sel_size}.xlsx")
            
            if st.button("🗑️ Reset để quét file mới"):
                st.session_state.audit_key += 1
                st.rerun()
    else:
        st.error("⚠️ Không đọc được thông số. Vui lòng kiểm tra lại trang POM trong PDF Reitmans.")
