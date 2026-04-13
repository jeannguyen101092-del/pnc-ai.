import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
# Lưu ý: Thay đổi URL và KEY thật của bạn vào đây
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

# Khởi tạo Supabase an toàn
try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("Chưa cấu hình Supabase URL/KEY!")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V43.7", page_icon="📊")

# Quản lý state để reset upload
if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

@st.cache_resource
def load_ai():
    # Sử dụng weights mới theo chuẩn Torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_ai()

# --- UTILS: XỬ LÝ SỐ & CHUẨN HÓA ---
def parse_reitmans_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none', 'null']: return 0
        # Regex tìm số thập phân hoặc phân số (1/2, 3/4...)
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def clean_pos(t):
    """Chuẩn hóa tên hạng mục để so sánh: bỏ khoảng trắng, viết hoa, bỏ ký tự đặc biệt"""
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- TRÍCH XUẤT DỮ LIỆU PDF ---
def extract_pom_new_v437(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"
    try:
        pdf_content = pdf_file.read()
        # 1. Lấy ảnh trang đầu làm Thumbnail
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
        
        all_text = ""
        for page in doc: all_text += (page.get_text() or "").upper() + " "
        if "REITMANS" in all_text: brand = "REITMANS"
        doc.close()

        # 2. Quét bảng thông số
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    
                    # Tìm dòng Header chứa POM NAME và NEW
                    p_name_idx, new_idx = -1, -1
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if any("POM NAME" in s for s in row_up) and any("NEW" in s for s in row_up):
                            # Tìm chính xác vị trí cột
                            for idx, val in enumerate(row_up):
                                if "POM NAME" in val: p_name_idx = idx
                                if "NEW" in val: new_idx = idx
                            
                            # Lấy dữ liệu từ các dòng sau header
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[p_name_idx]).replace('\n',' ').strip().upper()
                                if len(name) < 3 or any(x in name for x in ["REF:", "RELATED:", "TOTAL"]): continue
                                val_new = parse_reitmans_val(d_row[new_idx])
                                if val_new > 0: full_specs[name] = val_new
                            break
        return {"specs": full_specs, "img": img_bytes, "brand": brand}
    except Exception as e:
        st.error(f"Lỗi trích xuất: {e}")
        return None

# --- SIDEBAR: QUẢN LÝ KHO ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU MẪU")
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng mẫu trong kho", res_count.count if res_count.count else 0)
    except: st.warning("Chưa kết nối Database")

    files = st.file_uploader("Nạp Techpack mẫu (PDF)", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")
    
    if files and st.button("🚀 BẮT ĐẦU NẠP VÀO KHO"):
        prog = st.progress(0)
        for i, f in enumerate(files):
            # Kiểm tra trùng
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data: continue

            d = extract_pom_new_v437(f)
            if d and d['specs'] and d['img']:
                # Tạo Vector ảnh
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                tf = transforms.Compose([
                    transforms.Resize(224), transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                # Upload ảnh & Data
                path = f"lib_{f.name.replace('.pdf', '')}.png"
                supabase.storage.from_(BUCKET).upload(path=path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                
                img_url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": d['specs'],
                    "image_url": img_url, "category": d['brand']
                }).execute()
            prog.progress((i + 1) / len(files))
        st.success("Đã nạp xong!")
        st.session_state.up_key += 1
        st.rerun()

# --- MAIN: ĐỐI SOÁT ---
st.title("🔍 AI Fashion Auditor V43.7")
t_file = st.file_uploader("Upload file cần kiểm tra", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_new_v437(t_file)
    if target and target['specs']:
        st.info(f"Phân tích: Tìm thấy {len(target['specs'])} thông số. Đang so khớp với kho...")
        
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            # 1. Tính Vector file mới
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            # 2. So sánh Cosine Similarity
            matches = []
            for item in db_res.data:
                if item.get('vector'):
                    v_ref = np.array(item['vector']).reshape(1, -1)
                    sim_val = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": item, "sim": sim_val})
            
            # Lấy mẫu giống nhất
            if matches:
                best = sorted(matches, key=lambda x: x['sim'], reverse=True)[0]
                
                st.subheader(f"✨ Mẫu khớp nhất: {best['data']['file_name']} (Độ giống: {best['sim']:.1f}%)")
                
                col1, col2 = st.columns(2)
                with col1: st.image(target['img'], caption="Bản vẽ kiểm tra", use_container_width=True)
                with col2: st.image(best['data']['image_url'], caption="Mẫu gốc đối chiếu", use_container_width=True)

                # 3. So sánh bảng thông số
                diff_list = []
                ref_specs = best['data']['spec_json']
                
                # Tạo map chuẩn hóa để so khớp nhanh
                ref_map = {clean_pos(k): (k, v) for k, v in ref_specs.items()}

                for p_name, v_target in target['specs'].items():
                    p_clean = clean_pos(p_name)
                    v_ref = 0
                    if p_clean in ref_map:
                        v_ref = ref_map[p_clean][1]
                    
                    diff = round(v_target - v_ref, 3)
                    diff_list.append({
                        "Hạng mục (POM)": p_name,
                        "Bản đang kiểm": v_target,
                        "Mẫu gốc": v_ref,
                        "Chênh lệch": diff,
                        "Kết quả": "✅ OK" if abs(diff) < 0.001 else "❌ Lệch"
                    })
                
                if diff_list:
                    df_r = pd.DataFrame(diff_list)
                    
                    # Highlight lỗi
                    def highlight_diff(val):
                        if val == "❌ Lệch": return 'background-color: #ff4b4b; color: white'
                        return ''

                    st.table(df_r.style.applymap(highlight_diff, subset=['Kết quả']))
                    
                    # Xuất Excel
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df_r.to_excel(writer, index=False, sheet_name='Audit_Report')
                    
                    st.download_button("📥 Tải báo cáo Excel", out.getvalue(), "Audit_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            if st.button("🔄 Quét file mới"):
                st.session_state.au_key += 1
                st.rerun()
    else:
        st.error("Không trích xuất được dữ liệu từ PDF. Vui lòng kiểm tra lại định dạng file.")
