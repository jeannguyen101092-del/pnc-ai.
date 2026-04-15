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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96 Pro", page_icon="👖")

# ================= 2. HÀM AI & PHÂN LOẠI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# --- Cập nhật lại hàm classify để "tỉnh táo" hơn ---
def classify_garment(specs_dict):
    # Lấy toàn bộ tên thông số chuyển thành chữ hoa
    all_poms = " ".join(specs_dict.keys()).upper()
    
    # 1. QUẦN: Nếu thấy bất kỳ chữ nào liên quan đến đáy hoặc ống quần
    if any(k in all_poms for k in ["INSEAM", "OUTSEAM", "RISE", "LEG OPENING", "THIGH", "CROTCH"]):
        return "👖 QUẦN"
    
    # 2. ÁO: Nếu thấy vòng ngực và có liên quan đến tay áo
    if any(k in all_poms for k in ["BUST", "CHEST", "ARMHOLE", "SLEEVE", "SHOULDER"]):
        return "👕 ÁO / JACKET"
        
    return "👗 ĐẦM / KHÁC"

# --- Sửa đoạn hiển thị đối soát để tránh lỗi TypeError (Dòng 115-125) ---
if file_audit:
    target = extract_pdf_smart_scan(file_audit)
    if target and target["all_specs"]:
        # Lấy nhãn loại hàng từ size đầu tiên tìm thấy
        first_size = list(target['all_specs'].keys())[0]
        cat = classify_garment(target['all_specs'][first_size])
        st.info(f"📍 AI Nhận diện: **{cat}**")

        # ... (Phần tìm Top 3 giữ nguyên) ...

        # ĐOẠN FIX LỖI: Kiểm tra dữ liệu trước khi map
        sel_s = st.selectbox("Chọn Size:", list(target['all_specs'].keys()))
        d_audit = target['all_specs'].get(sel_s, {})
        
        # Lấy dữ liệu mẫu trong thư viện
        lib_specs_all = best.get('spec_json', {})
        # Tìm size khớp, nếu không có lấy size đầu tiên có trong kho của mẫu đó
        s_lib = sel_s if sel_s in lib_specs_all else (list(lib_specs_all.keys())[0] if lib_specs_all else None)
        
        if s_lib:
            d_lib = lib_specs_all[s_lib]
            report = []
            for pom, val in d_audit.items():
                # Dùng .get(pom, 0) để nếu mẫu kho thiếu thông số đó thì vẫn không bị lỗi crash
                ref = d_lib.get(pom, 0) 
                diff = round(val - ref, 4)
                report.append({
                    "Thông số": pom, 
                    "Thực tế": val, 
                    "Mẫu kho": ref if ref != 0 else "N/A", 
                    "Lệch": diff if ref != 0 else 0,
                    "Kết quả": "✅ OK" if (ref != 0 and abs(diff) <= 0.25) else "⚠️ Ko đối soát"
                })
            
            df_rep = pd.DataFrame(report)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)
        else:
            st.error("Mẫu trong kho không có dữ liệu size để đối soát.")


# ================= 3. HÀM QUÉT PDF THÔNG MINH (CHỈ QUÉT TRANG POM) =================
def extract_pdf_smart_scan(file):
    all_specs, img_bytes = {}, None
    try:
        file.seek(0); pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        
        # Tìm trang chứa bảng (POM, Measurement, Spec...)
        target_pages = [0] # Ưu tiên trang 1
        for i in range(len(doc)):
            text = doc[i].get_text().upper()
            if any(k in text for k in ["POM", "MEASUREMENT", "SPEC", "DIMENSION", "WAIST", "INSEAM"]):
                if i not in target_pages: target_pages.append(i)
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for p_idx in target_pages:
                page = pdf.pages[p_idx]
                tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 5})
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 3: continue
                    
                    desc_col, size_cols = -1, {}
                    for r_idx in range(min(15, len(df))):
                        row = [str(c).strip().upper() for c in df.iloc[r_idx]]
                        for i, v in enumerate(row):
                            if any(x in v for x in ["POM", "DESCRIPTION", "POSITION", "NAME"]): desc_col = i; break
                        for i, v in enumerate(row):
                            if i == desc_col or not v: continue
                            if v.isdigit() or v in ["XS","S","M","L","XL","XXL","3XL"]:
                                if not any(x in v for x in ["TOL", "+/-"]): size_cols[i] = v
                        if desc_col != -1 and size_cols: break
                    
                    if desc_col != -1:
                        for s_col, s_name in size_cols.items():
                            if s_name not in all_specs: all_specs[s_name] = {}
                            for d_idx in range(len(df)):
                                pom = str(df.iloc[d_idx, desc_col]).replace('\n', ' ').strip()
                                if len(pom) > 3 and not (pom.isupper() and len(pom) > 25):
                                    val = parse_val(df.iloc[d_idx, s_col])
                                    if val > 0: all_specs[s_name][pom.upper()] = val
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. GIAO DIỆN & LUỒNG XỬ LÝ =================
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0

with st.sidebar:
    st.header("🏢 KHO MẪU")
    res_db = supabase.table("ai_data").select("id", "file_name", count="exact").execute()
    st.metric("Tổng tồn kho", f"{res_db.count if res_db.count else 0} mẫu")
    existing_files = [x['file_name'] for x in res_db.data] if res_db.data else []
    
    files = st.file_uploader("Nạp mẫu mới", accept_multiple_files=True, key=str(st.session_state['uploader_key']))
    if files and st.button("NẠP KHO"):
        for f in files:
            if f.name in existing_files:
                st.warning(f"Bỏ qua: {f.name} đã tồn tại."); continue
            data = extract_pdf_smart_scan(f)
            if data and data['all_specs']:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.session_state['uploader_key'] += 1
        st.rerun()

st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Upload PDF Audit", type="pdf")

if file_audit:
    # --- ĐÃ SỬA LỖI NAMEERROR TẠI ĐÂY ---
    target = extract_pdf_smart_scan(file_audit) 
    
    if target and target["all_specs"]:
        cat = classify_garment(next(iter(target['all_specs'].values())))
        st.info(f"📍 AI Nhận diện: **{cat}**")

        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            top_3 = df_db.sort_values('sim', ascending=False).head(3)
            
            st.subheader("🎯 CHỌN MẪU ĐỐI SOÁT")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], use_container_width=True)
                    if st.button(f"Mẫu {i+1}: {row['sim']:.1%}", key=f"sel_{i}"):
                        st.session_state['active_idx'] = idx
            
            best = top_3.loc[st.session_state.get('active_idx', top_3.index)]
            st.divider()
            
            sel_s = st.selectbox("Chọn Size:", list(target['all_specs'].keys()))
            d_audit = target['all_specs'][sel_s]
            d_lib = best['spec_json'].get(sel_s, list(best['spec_json'].values())[0])
            
            report = []
            for pom, val in d_audit.items():
                ref = d_lib.get(pom, 0)
                diff = round(val - ref, 4)
                report.append({"Thông số": pom, "Thực tế": val, "Mẫu kho": ref, "Chênh lệch": diff, "Kết quả": "✅ OK" if abs(diff) <= 0.25 else "❌ LỆCH"})
            
            df_rep = pd.DataFrame(report)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)
            
            output = io.BytesIO()
            df_rep.to_excel(output, index=False, engine='xlsxwriter')
            st.download_button("📥 TẢI EXCEL", output.getvalue(), f"Report_{sel_s}.xlsx")
    else:
        st.error("⚠️ Không tìm thấy bảng thông số.")
