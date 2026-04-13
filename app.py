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

st.set_page_config(layout="wide", page_title="AI V20.0 - PRO V76", page_icon="🛡️")

# ================= 2. MODEL AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= 3. HÀM PHÂN LOẠI THÔNG MINH =================
def detect_category(text):
    """Phân tích nội dung text để nhận diện loại hàng"""
    t = text.upper()
    if any(x in t for x in ["PANT", "JEAN", "BOTTOM", "SHORT", "TROUSER", "LEGGING"]): return "QUẦN"
    if any(x in t for x in ["SHIRT", "TOP", "BLOUSE", "TEE", "JACKET", "HOODIE", "SWEATER"]): return "ÁO"
    if any(x in t for x in ["DRESS", "SKIRT", "GOWN", "JUMPSUIT"]): return "VÁY/ĐẦM"
    return "KHÁC"

# ================= 4. HÀM TRÍCH XUẤT CẢI TIẾN (VỚI ĐIỀU KIỆN LỌC) =================
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

def extract_pdf_v76(file):
    specs, img_bytes, full_text = {}, None, ""
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
            for page in doc: full_text += page.get_text()
        doc.close()
        
        # Nhận diện chủng loại
        category = detect_category(full_text)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2: continue
                    n_col, v_col = -1, -1
                    for r_idx, row in df.head(15).iterrows():
                        row_up = [str(c).upper().strip() for c in row if c]
                        if any(x in " ".join(row_up) for x in ["DESCRIPTION", "DESC"]):
                            for i, v in enumerate(row_up):
                                if "DESCRIPTION" in v or "DESC" in v: n_col = i; break
                            for i, v in enumerate(row_up):
                                if any(target in v for target in ["NEW", "SAMPLE", "SPEC", "32", "34"]):
                                    v_col = i; break
                            if n_col != -1 and v_col != -1:
                                for d_idx in range(r_idx + 1, len(df)):
                                    d_row = df.iloc[d_idx]
                                    name = str(d_row[n_col]).replace('\n', ' ').strip().upper()
                                    if len(name) < 3 or any(x in name for x in ["TOL", "REF"]): continue
                                    val = parse_val(d_row[v_col])
                                    if val > 0: specs[name] = val
                                break
                if specs: break
        
        # ĐIỀU KIỆN LOẠI BỎ: Ít nhất 5 thông số và có ảnh
        if len(specs) < 5 or img_bytes is None:
            return None
            
        return {"specs": specs, "img": img_bytes, "category": category}
    except: return None

# ================= 5. SIDEBAR (CHỐNG TRÙNG LẶP & LỌC FILE) =================
with st.sidebar:
    st.header("🛡️ AI V20.0 - PRO V76")
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    st.info(f"📁 Kho mẫu: {res_count.count if res_count.count else 0} file")
    
    st.divider()
    st.subheader("🚀 NẠP KHO THÔNG MINH")
    new_files = st.file_uploader("Upload PDF nạp kho", type="pdf", accept_multiple_files=True)
    
    if new_files and st.button("XÁC NHẬN NẠP KHO"):
        for f in new_files:
            # CHỐNG TRÙNG LẶP: Kiểm tra tên file trong DB
            check = supabase.table("ai_data").select("file_name").eq("file_name", f.name).execute()
            if check.data:
                st.warning(f"⏩ Bỏ qua: `{f.name}` đã tồn tại trong kho.")
                continue

            with st.spinner(f"Đang phân tích {f.name}..."):
                data = extract_pdf_v76(f)
                
                # LOẠI BỎ FILE KHÔNG ĐỦ ĐIỀU KIỆN
                if not data:
                    st.error(f"❌ Loại bỏ: `{f.name}` rác hoặc không đủ thông số/hình ảnh.")
                    continue
                
                # Nạp kho nếu hợp lệ
                img = Image.open(io.BytesIO(data['img'])).convert('RGB')
                tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                vec = model_ai(tf(img).unsqueeze(0)).flatten().detach().cpu().numpy().astype(float).tolist()
                path = f"lib_{re.sub(r'[^a-zA-Z0-9]', '_', f.name.replace('.pdf',''))}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true", "content-type": "image/png"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)
                
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": vec, "spec_json": data['specs'], 
                    "image_url": url, "category": data['category'] # Lưu chủng loại
                }).execute()
                st.toast(f"✅ Đã nạp {f.name} ({data['category']})")
        st.rerun()

# ================= 6. MAIN (SO SÁNH CÙNG CHỦNG LOẠI) =================
st.subheader("📊 SMART PRODUCT COMPARISON")
file_audit = st.file_uploader("Upload file đối soát", type="pdf", label_visibility="collapsed")

if file_audit:
    target = extract_pdf_v76(file_audit)
    if target:
        st.success(f"✨ Phân tích: Loại hàng **{target['category']}** | {len(target['specs'])} hạng mục.")
        
        # Chỉ lấy dữ liệu CÙNG CATEGORY từ Database
        db_all = supabase.table("ai_data").select("*").eq("category", target['category']).execute()
        
        if not db_all.data:
            st.warning(f"⚠️ Trong kho chưa có mẫu nào thuộc loại **{target['category']}** để so sánh.")
        else:
            # Tạo vector kiểm tra
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1).astype(np.float32)
            
            matches = []
            for item in db_all.data:
                v_ref = np.array(item["vector"]).reshape(1, -1).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref)[0][0])
                matches.append({"item": item, "score": score})
            
            best_match = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
            best = best_match['item']
            
            # --- HIỂN THỊ SONG SONG ( Side-by-side) ---
            col1, col2 = st.columns(2)
            with col1:
                st.info("📄 BẢN ĐANG KIỂM TRA")
                st.image(target["img"], use_container_width=True)
                st.table(pd.DataFrame([{"Hạng mục": k, "Số đo": v} for k,v in target["specs"].items()]))
            
            with col2:
                st.success(f"✨ MẪU GỐC (Cùng loại: {best['category']} | Khớp {best_match['score']*100:.1f}%)")
                st.image(best['image_url'], use_container_width=True)
                ref_specs = best['spec_json']
                rows = []
                clean_ref = {re.sub(r'[^A-Z0-9]', '', k.upper()): v for k, v in ref_specs.items()}
                for k, v in target["specs"].items():
                    k_c = re.sub(r'[^A-Z0-9]', '', k.upper())
                    v_ref = clean_ref.get(k_c, 0)
                    diff = round(v - v_ref, 3)
                    rows.append({"Vị trí": k, "Mới": v, "Kho": v_ref, "Kết quả": "Khớp" if abs(diff) < 0.1 else "Lệch"})
                
                df_res = pd.DataFrame(rows)
                st.table(df_res.style.map(lambda x: 'color: green; font-weight: bold' if x == 'Khớp' else 'color: red; font-weight: bold', subset=['Kết quả']))

            # NÚT XUẤT EXCEL
            st.divider()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Audit_Report')
            st.download_button(label="📥 TẢI BÁO CÁO EXCEL", data=output.getvalue(), file_name=f"Audit_{best['file_name']}.xlsx", mime="application/vnd.ms-excel")
    else:
        st.error("❌ File không đủ điều kiện đối soát (Thiếu bảng thông số hoặc hình ảnh).")
