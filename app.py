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

st.set_page_config(layout="wide", page_title="AI V20.0 - PRO V79 Master", page_icon="🛡️")

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= 3. PHÂN LOẠI THÔNG MINH =================
def detect_category(text):
    t = str(text).upper()
    if any(x in t for x in ["PANT", "JEAN", "BOTTOM", "SHORT", "TROUSER", "LEGGING"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "BLOUSE", "TEE", "JACKET", "HOODIE", "SWEATER"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "JUMPSUIT"]): return "VÁY/ĐẦM"
    return "KHÁC"

# ================= 4. HÀM TRÍCH XUẤT MASTER V79 =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        # Xử lý các dạng 1 1/2, 23.5, 12...
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0] # Lấy kết quả đầu tiên
        if ' ' in v:
            parts = v.split()
            return float(parts[0]) + eval(parts[1])
        if '/' in v: return eval(v)
        return float(v)
    except: return 0

def extract_pdf_v79(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        
        # 1. Lấy ảnh trang đầu
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += (page.get_text() or "")
        doc.close()
        
        category = detect_category(full_text)

        # 2. Quét bảng (Nới lỏng điều kiện tìm Header)
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).dropna(how='all', axis=1)
                    if df.empty or len(df.columns) < 2: continue
                    
                    n_col, v_col = -1, -1
                    # Tìm Header thông minh: Quét 25 dòng đầu
                    for r_idx, row in df.head(25).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        
                        # TÌM CỘT DESCRIPTION (Tìm kiếm mờ)
                        for i, v in enumerate(row_up):
                            if any(x in v for x in ["DESC", "POM NAME", "MEASUREMENT", "POSITION"]):
                                n_col = i; break
                        
                        # TÌM CỘT GIÁ TRỊ (Tìm kiếm mờ các cột Spec/Size)
                        for i, v in enumerate(row_up):
                            if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "SIZE", "30", "32", "34", "M", "S", "L"]):
                                if i != n_col: # Tránh trùng với cột tên
                                    v_col = i; break
                        
                        if n_col != -1 and v_col != -1:
                            for d_idx in range(r_idx + 1, len(df)):
                                d_row = df.iloc[d_idx]
                                name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                # Bỏ qua các dòng tiêu đề phụ rỗng hoặc dòng dung sai
                                if len(name) < 3 or any(x in name for x in ["TOL", "REF", "REMARK"]): continue
                                val = parse_val(d_row[v_col])
                                if val > 0: specs[name] = val
                            break
                if specs: break 
        
        # BỎ BỘ LỌC KHẮT KHE: Chỉ cần có bảng là nạp
        if not specs: return None
        return {"specs": specs, "img": img_bytes, "category": category}
    except: return None

# ================= 5. SIDEBAR =================
with st.sidebar:
    st.header("🛡️ AI V20.0 - PRO V79")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.info(f"📁 Kho mẫu hiện tại: {res_count.count if res_count.count else 0} file")
    
    st.divider()
    st.subheader("🚀 NẠP KHO THÔNG MINH")
    new_files = st.file_uploader("Upload PDF nạp kho", type="pdf", accept_multiple_files=True)
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            # Chống trùng
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data:
                st.warning(f"⏩ Đã có: {f.name}"); continue

            with st.spinner(f"Đang xử lý {f.name}..."):
                data = extract_pdf_v79(f)
                if not data:
                    st.error(f"❌ Không thể đọc bảng trong file `{f.name}`. Hãy kiểm tra PDF."); continue
                
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                supabase.table("ai_data").insert({"file_name": f.name, "vector": vec, "spec_json": data['specs'], "image_url": url, "category": data['category']}).execute()
                st.toast(f"✅ Đã nạp {f.name} ({data['category']})")
        st.rerun()

    if st.button("🗑️ Dọn dẹp kho"):
        supabase.table("ai_data").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        st.rerun()

# ================= 6. MAIN =================
st.subheader("📊 SMART PRODUCT COMPARISON")
file_audit = st.file_uploader("Upload file đối soát", type="pdf", label_visibility="collapsed")

if file_audit:
    with st.spinner("Đang trích xuất dữ liệu..."):
        target = extract_pdf_v79(file_audit)
    
    if target:
        st.success(f"✨ Phát hiện loại hàng: **{target['category']}** | {len(target['specs'])} thông số.")
        
        db_res = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if not db_res.data:
            st.warning(f"⚠️ Chưa có mẫu nào cùng loại **{target['category']}** trong kho mẫu.")
        else:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
            
            matches = []
            for item in db_res.data:
                if item.get("vector") and len(item["vector"]) == 512:
                    v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                    score = float(cosine_similarity(v_test, v_ref)[0][0]) # Lấy giá trị chính xác
                    matches.append({"item": item, "score": score})
            
            if matches:
                top = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
                best = top['item']
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("📄 BẢN ĐANG KIỂM TRA")
                    st.image(target["img"], use_container_width=True)
                    st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))
                
                with c2:
                    st.success(f"✨ MẪU GỐC (Khớp {top['score']*100:.1f}%)")
                    st.image(best['image_url'], use_container_width=True)
                    ref_specs = best['spec_json']
                    rows = []
                    clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): v for k, v in ref_specs.items()}
                    for k, v in target["specs"].items():
                        k_c = re.sub(r'[^A-Z0-9]', '', k.upper())
                        v_ref = clean_ref.get(k_c, 0)
                        diff = round(v - v_ref, 3)
                        rows.append({"Thông số": k, "Mới": v, "Mẫu Gốc": v_ref, "Kết quả": "Khớp" if abs(diff) < 0.1 else "Lệch"})
                    
                    df_res = pd.DataFrame(rows)
                    st.table(df_res.style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))
                
                # Xuất Excel
                st.divider()
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False)
                st.download_button("📥 TẢI BÁO CÁO EXCEL", out.getvalue(), f"Audit_{best['file_name']}.xlsx")
    else:
        st.error("❌ Không tìm thấy bảng thông số hợp lệ. Hãy kiểm tra PDF của bạn có phải dạng Text không.")
