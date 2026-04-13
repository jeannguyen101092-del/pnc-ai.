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

st.set_page_config(layout="wide", page_title="AI V20.0 - PRO V73", page_icon="🛡️")

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= 3. HÀM XỬ LÝ DỮ LIỆU CẢI TIẾN =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_pdf_v73(file):
    specs, img_bytes = {}, None
    try:
        file.seek(0)
        pdf_content = file.read()
        # Chụp ảnh trang 1 làm thumbnail
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            # Quét TOÀN BỘ các trang để không bỏ sót bảng thông số
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(axis=1, how='all')
                    if df.empty or len(df.columns) < 2: continue
                    
                    n_col, v_col = -1, -1
                    # Tìm Header thông minh: Quét 20 dòng đầu của mỗi bảng
                    for r_idx, row in df.head(20).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        row_str = " ".join(row_up)
                        
                        # TÌM CỘT TÊN (DESCRIPTION): Ưu tiên Description, Desc, POM Name
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESCRIPTION", "DESC", "POM NAME", "MEASUREMENT ITEM"]):
                                n_col = i; break
                        
                        # Nếu không thấy Description, thử tìm cột POM mã số
                        if n_col == -1:
                            for i, v in enumerate(row_up):
                                if "POM" in v: n_col = i; break
                        
                        # TÌM CỘT GIÁ TRỊ: Tìm NEW, SAMPLE, hoặc các cột có số Size (30, 32, 34...)
                        for i, v in enumerate(row_up):
                            if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "30", "32", "34", "36", "38"]):
                                v_col = i; break
                        
                        # Nếu đã xác định được cả 2 cột
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                # Loại bỏ các dòng tiêu đề rác
                                if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARK", "COMMENTS"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: specs[name] = val
                            break # Thoát vòng lặp row nếu đã lấy xong dữ liệu bảng này
                if specs: break # Nếu đã có dữ liệu từ trang này thì không cần quét trang sau
        return {"specs": specs, "img": img_bytes}
    except Exception as e:
        st.error(f"Lỗi xử lý file: {e}")
        return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("🛡️ AI V20.0 - PRO V73")
    try:
        res_count = supabase.table("ai_data").select("id", count="exact").execute()
        st.info(f"📁 Kho mẫu: {res_count.count if res_count.count else 0} file")
    except: st.error("Chưa kết nối Supabase")
    
    st.divider()
    st.subheader("🚀 NẠP TECHPACK MỚI")
    new_files = st.file_uploader("Upload PDF nạp kho", type="pdf", accept_multiple_files=True)
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            with st.spinner(f"Đang phân tích: {f.name}..."):
                data = extract_pdf_v73(f)
                if data and data['specs'] and data['img']:
                    # 1. Tạo Vector AI
                    img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                    vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                    
                    # 2. Lưu vào Supabase
                    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))
                    path = f"lib_{safe_name}.png"
                    
                    try:
                        supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                        url = supabase.storage.from_(BUCKET).get_public_url(path)
                        supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url}).execute()
                        st.toast(f"✅ Đã nạp: {f.name}")
                    except Exception as e: st.error(f"Lỗi lưu trữ: {e}")
                else:
                    st.warning(f"⚠️ Không tìm thấy bảng thông số trong file: {f.name}")
        st.rerun()

    if st.button("🗑️ Dọn dẹp kho"):
        supabase.table("ai_data").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        st.rerun()

# ================= 5. MAIN (SO SÁNH) =================
st.subheader("📊 PRODUCT SUMMARY COMPARISON")
file_audit = st.file_uploader("Upload file cần đối soát", type="pdf", label_visibility="collapsed")

if file_audit:
    with st.spinner("Đang trích xuất dữ liệu đối soát..."):
        target = extract_pdf_v73(file_audit)
    
    if target and target["specs"]:
        st.success(f"✨ Tìm thấy {len(target['specs'])} hạng mục thông số.")
        
        # Logic so sánh (Giống V71 nhưng dùng extract mới)
        db_all = supabase.table("ai_data").select("*").execute()
        if db_all.data:
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
                top = sorted(matches, key=lambda x: x['score'], reverse=True)
                best = top[0]['item']
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("📄 BẢN ĐANG KIỂM TRA")
                    st.image(target["img"], use_container_width=True)
                    st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))
                
                with c2:
                    st.success(f"✨ MẪU GỐC TRONG KHO ({top[0]['score']*100:.1f}%)")
                    st.image(best['image_url'], use_container_width=True)
                    ref_specs = best['spec_json']
                    rows = []
                    clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): v for k, v in ref_specs.items()}
                    for k, v in target["specs"].items():
                        k_c = re.sub(r'[^A-Z0-9]', '', k.upper())
                        v_ref = clean_ref.get(k_c, 0)
                        diff = round(v - v_ref, 3)
                        rows.append({"Thông số": k, "Mới": v, "Kho mẫu": v_ref, "Kết quả": "Khớp" if abs(diff) < 0.1 else "Lệch"})
                    st.table(pd.DataFrame(rows).style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold' if x == 'Lệch' else '', subset=['Kết quả']))
    else:
        st.error("❌ Không trích xuất được bảng thông số. Hãy kiểm tra xem file PDF có phải là bản quét (Scan ảnh) không.")
