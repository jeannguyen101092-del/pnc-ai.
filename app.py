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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96", page_icon="👖")

# ================= 2. HÀM AI & PHÂN LOẠI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def classify_garment(specs_dict):
    """Nhận diện loại hàng dựa trên từ khóa thông số"""
    text = " ".join(specs_dict.keys()).upper()
    if any(k in text for k in ["INSEAM", "OUTSEAM", "THIGH", "LEG OPENING"]):
        return "QUẦN (PANTS/SHORTS)"
    if any(k in text for k in ["BUST", "CHEST", "SHOULDER", "NECK"]):
        if any(k in text for k in ["SLEEVE"]): return "ÁO (TOP/JACKET)"
        return "ÁO KHÔNG TAY/VÁY"
    return "CHƯA XÁC ĐỊNH"

def parse_val(t_top, t_bottom=""):
    """Kết hợp dòng trên (số nguyên) và dòng dưới (phân số)"""
    try:
        def clean_num(txt):
            txt = str(txt).replace(',', '.').strip().lower()
            return re.sub(r'[^\d\./\s]', '', txt)

        v1 = clean_num(t_top)
        v2 = clean_num(t_bottom)
        
        total = 0.0
        # Xử lý số nguyên ở dòng trên
        if v1 and v1.replace('.','').isdigit():
            total += float(v1)
        
        # Xử lý phân số ở dòng dưới (ví dụ: 3/4)
        if "/" in v2:
            parts = v2.split('/')
            if len(parts) == 2:
                total += float(parts[0]) / float(parts[1])
        elif v2 and v2.replace('.','').isdigit():
            total += float(v2)
            
        return total
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

# ================= 3. TRÍCH XUẤT PDF =================
def extract_pdf_smart_scan(file):
    all_specs, img_bytes = {}, None
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        
        # 1. TÌM TRANG THÔNG SỐ (Bất kể tên là gì)
        target_pages = []
        for i in range(len(doc)):
            text = doc[i].get_text().upper()
            # Danh sách từ khóa mở rộng (Càng nhiều càng tốt)
            keywords = ["POM", "MEASUREMENT", "SPEC", "DIMENSION", "SIZE CHART", "TOLERANCE", "WAIST", "INSEAM"]
            if any(k in text for k in keywords):
                target_pages.append(i)
        doc.close()

        # 2. QUÉT NHANH
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for p_idx in target_pages:
                page = pdf.pages[p_idx]
                # Cấu hình quét bảng linh hoạt cho cả bảng có khung và không khung
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text", 
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5, # Tăng độ nhạy để gom các chữ gần nhau
                })
                
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 3: continue
                    
                    # Tự động nhận diện cột Tên thông số và cột Size
                    desc_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        
                        for i, v in enumerate(row):
                            # Tìm cột chứa tên vị trí đo
                            if any(x in v for x in ["POM", "DESCRIPTION", "NAME", "POSITION"]):
                                desc_col = i; break
                        
                        for i, v in enumerate(row):
                            if i == desc_col or not v: continue
                            # Tìm các cột Size (Số hoặc Chữ)
                            if v.isdigit() or v in ["XS","S","M","L","XL","XXL","3XL"]:
                                if not any(x in v for x in ["TOL", "+/-", "DATE"]):
                                    size_cols[i] = v
                        if desc_col != -1 and size_cols: break
                    
                    if desc_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                # Chỉ lấy các dòng có tên thực tế, bỏ qua dòng tiêu đề HOA TOÀN BỘ của khách
                                if len(pom) > 3 and not (pom.isupper() and len(pom) > 25):
                                    val = parse_val(df.iloc[d_idx, s_col])
                                    if val > 0:
                                        all_specs[s_name][pom.upper()] = val
                                        
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None


# ================= 4. GIAO DIỆN CHÍNH =================
# --- Thêm biến vào đầu chương trình để quản lý việc reset file ---
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = 0

with st.sidebar:
    st.header("🏢 KHO MẪU")
    # Hiển thị số lượng mẫu
    res_count = supabase.table("ai_data").select("id", "file_name", count="exact").execute()
    count = res_count.count if res_count.count else 0
    existing_files = [item['file_name'] for item in res_count.data] if res_count.data else []
    st.metric("Tổng tồn kho", f"{count} mẫu")
    
    # Sử dụng key động để reset file uploader
    new_files = st.file_uploader(
        "Nạp mẫu mới", 
        accept_multiple_files=True, 
        key=str(st.session_state['file_uploader_key'])
    )

    if new_files and st.button("NẠP KHO"):
        process_count = 0
        for f in new_files:
            if f.name in existing_files:
                st.warning(f"⚠️ Bỏ qua: {f.name} đã tồn tại.")
                continue
            
            data = extract_pdf_multi_size(f)
            if data:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                # Upload ảnh và dữ liệu lên Supabase
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, 
                    "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], 
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
                process_count += 1
        
        if process_count > 0:
            st.success(f"✅ Đã nạp thành công {process_count} mẫu!")
            
            # --- ĐÂY LÀ PHẦN QUAN TRỌNG ĐỂ XÓA FILE ---
            st.session_state['file_uploader_key'] += 1 # Thay đổi key để Streamlit vẽ lại uploader trống
            st.rerun() # Load lại trang để xóa sạch danh sách file cũ


st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    target = extract_pdf_multi_size(file_audit)
    if target and target["all_specs"]:
        # Nhận diện loại hàng
        sample_specs = list(target['all_specs'].values())[0]
        category = classify_garment(sample_specs)
        st.info(f"📍 Loại hàng nhận diện: **{category}**")

        # So sánh tìm TOP 3
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            
            st.subheader("🎯 TOP 3 MẪU TƯƠNG ĐỒNG NHẤT")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], caption=f"{row['file_name']} ({row['sim']:.1%})")
                    if st.button(f"Chọn mẫu {i+1}", key=f"btn_{i}"): st.session_state['selected_idx'] = idx

            # Mặc định chọn mẫu cao nhất hoặc theo nút bấm
            best = top_3.loc[st.session_state.get('selected_idx', top_3.index[0])]
            st.divider()
            st.subheader(f"📊 ĐỐI SOÁT CHI TIẾT: {best['file_name']}")
            
            sel_size = st.selectbox("Chọn Size đối soát:", list(target['all_specs'].keys()))
            spec_audit = target['all_specs'][sel_size]
            spec_ref = best['spec_json'].get(sel_size, list(best['spec_json'].values())[0])
            
            report = []
            for pom, val in spec_audit.items():
                ref_val = spec_ref.get(pom, 0)
                diff = round(val - ref_val, 3)
                report.append({
                    "Thông số (Description)": pom, "Thực tế": val, "Mẫu kho": ref_val, 
                    "Chênh lệch": diff, "Kết quả": "✅ OK" if abs(diff) < 0.25 else "❌ LỆCH"
                })
            
            df_rep = pd.DataFrame(report)
            st.dataframe(df_rep.style.highlight_max(subset=['Chênh lệch'], color='#ff4b4b00'), use_container_width=True)
            
            towrite = io.BytesIO()
            df_rep.to_excel(towrite, index=False, engine='xlsxwriter')
            st.download_button(label="📥 TẢI BÁO CÁO EXCEL", data=towrite.getvalue(), file_name=f"Audit_{best['file_name']}.xlsx")
    else:
        st.error("Không tìm thấy dữ liệu thông số.")
