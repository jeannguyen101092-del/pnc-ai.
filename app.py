import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import os

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="PPJ GROUP | AI Auditor Pro", page_icon="👔")

if 'reset_key' not in st.session_state: st.session_state['reset_key'] = 0

# ================= 2. BỘ MÁY AI =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_file_hash(file_bytes): return hashlib.md5(file_bytes).hexdigest()

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): 
        return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def parse_val(t):
    try:
        if not t or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds|tol|grade)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return float(eval(v)) if '/' in v else float(v)
    except: return 0

# ================= 3. TRÍCH XUẤT PDF =================
def extract_pdf_full_logic(file_content):
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_temp = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO()
        img_temp.save(buf, format="WEBP", quality=85)
        img_bytes = buf.getvalue()
        doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for p in pdf.pages:
                tables = p.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if df.empty or len(df.columns) < 2: continue
                    d_col = df.apply(lambda x: x.astype(str).str.len().mean()).idxmax()
                    for col_idx in range(len(df.columns)):
                        if col_idx == d_col: continue
                        if sum([parse_val(v) for v in df.iloc[:, col_idx].head(10)]) > 0:
                            s_name = str(df.iloc[0, col_idx]).strip().replace('\n', ' ')
                            temp_data = {str(df.iloc[d, d_col]).strip(): parse_val(df.iloc[d, col_idx]) for d in range(len(df)) if len(str(df.iloc[d, d_col])) > 2}
                            if temp_data:
                                if s_name not in all_specs: all_specs[s_name] = {}
                                all_specs[s_name].update(temp_data)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. HIỂN THỊ SO SÁNH =================
def render_comparison_logic(target_data, repo_data):
    st.divider()
    st.subheader("📊 GIAI ĐOẠN 1: SO SÁNH HÌNH ẢNH")
    v_c1, v_c2 = st.columns(2)
    with v_c1: st.image(repo_data['image_url'], caption="BẢN GỐC (REPO)", use_container_width=True)
    with v_c2: st.image(target_data['img'], caption="BẢN MỚI (UPLOAD)", use_container_width=True)

    st.subheader("📊 GIAI ĐOẠN 2: ĐỐI SOÁT THÔNG SỐ")
    t_specs, r_specs = target_data['all_specs'], repo_data.get('spec_json', {})
    
    # Tìm bảng trùng tên
    common_tables = list(set(t_specs.keys()).intersection(set(r_specs.keys())))
    
    if common_tables:
        sel_tb = st.selectbox("Chọn bảng thông số để so sánh:", common_tables)
        t_d, r_d = t_specs[sel_tb], r_specs[sel_tb]
    else:
        st.warning("⚠️ Không tìm thấy bảng trùng tên. Đang gộp dữ liệu để đối soát...")
        # Gộp tất cả thông số lại nếu không khớp tên bảng
        t_d = {k: v for d in t_specs.values() for k, v in d.items()}
        r_d = {k: v for d in r_specs.values() for k, v in d.items()}

    pts = sorted(list(set(t_d.keys()).intersection(set(r_d.keys()))))
    
    if pts:
        comp_rows = []
        for p in pts:
            diff = round(t_d[p] - r_d[p], 3)
            status = "✅ KHỚP" if abs(diff) <= 0.125 else ("❌ LỆCH" if abs(diff) >= 0.5 else "⚠️ NHẸ")
            comp_rows.append({"Điểm đo": p, "Bản Mới (B)": t_d[p], "Bản Gốc (A)": r_d[p], "Chênh lệch": diff, "Trạng thái": status})
        
        df_comp = pd.DataFrame(comp_rows)
        
        # NÚT XUẤT EXCEL
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_comp.to_excel(writer, index=False, sheet_name='Audit')
        st.download_button("📥 Xuất báo cáo Excel", output.getvalue(), f"Audit_Result.xlsx")
        
        st.dataframe(df_comp.style.applymap(lambda v: 'background-color: #ffcccc' if v=="❌ LỆCH" else ('background-color: #fff3cd' if v=="⚠️ NHẸ" else ''), subset=['Trạng thái']), use_container_width=True)
    else:
        st.error("❌ Không tìm thấy các điểm đo (Points of Measure) tương ứng để so sánh.")

# ================= 5. THANH SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Tổng số mẫu trong kho", f"{count}")
    
    used_mb = count * 0.08
    st.write(f"💾 **Bộ nhớ:** {used_mb:.1f}MB / 1024MB")
    st.progress(min((used_mb / 1024), 1.0))
    st.divider()
    
    new_files = st.file_uploader("Tải file lên kho dữ liệu", accept_multiple_files=True, key=f"up_{st.session_state['reset_key']}")
    if new_files and st.button("ĐỒNG BỘ VÀO KHO", use_container_width=True):
        for f in new_files:
            fb = f.read(); h = get_file_hash(fb)
            data = extract_pdf_full_logic(fb)
            if data:
                path = f"lib_{h}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({"id": h, "file_name": f.name, "vector": get_image_vector(data['img']), "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)}).execute()
        st.rerun()
    if st.button("XÓA DANH SÁCH TẠM", use_container_width=True): st.session_state['reset_key'] += 1; st.rerun()

# ================= 6. GIAO DIỆN CHÍNH =================
st.title("👔 AI SMART AUDITOR PRO")
mode = st.radio("Chế độ:", ["Tìm kiếm AI (Audit)", "So sánh 2 phiên bản (Control)"], horizontal=True)

if mode == "Tìm kiếm AI (Audit)":
    f_audit = st.file_uploader("Tải file cần kiểm tra:", type="pdf")
    if f_audit:
        target = extract_pdf_full_logic(f_audit.read())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            db = pd.DataFrame(res.data)
            db['sim'] = cosine_similarity(np.array(get_image_vector(target['img'])).reshape(1, -1), np.array([v for v in db['vector']])).flatten()
            top = db.sort_values('sim', ascending=False).head(3)
            
            st.subheader("🎯 Gợi ý mẫu tương đồng nhất")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(top.iterrows()):
                with cols[i]:
                    st.image(row['image_url'], caption=f"Độ khớp: {row['sim']:.1%}")
                    if st.button(f"ĐỐI SOÁT VỚI MẪU {i+1}", key=f"s_{idx}"): st.session_state['selected_item'] = row.to_dict()
            
            if 'selected_item' in st.session_state:
                render_comparison_logic(target, st.session_state['selected_item'])

else:
    st.subheader("🔄 So sánh bản cũ (Trong kho) vs bản mới (Tải lên)")
    res = supabase.table("ai_data").select("file_name", "image_url", "spec_json").execute()
    repo_dict = {item['file_name']: item for item in res.data}
    
    col_a, col_b = st.columns(2)
    with col_a:
        sel_name = st.selectbox("1. Chọn bản gốc (Round A):", ["-- Chọn mẫu --"] + list(repo_dict.keys()))
        if sel_name != "-- Chọn mẫu --":
            st.image(repo_dict[sel_name]['image_url'], use_container_width=True)

    with col_b:
        f_new = st.file_uploader("2. Tải lên bản mới (Round B):", type="pdf")
        if f_new:
            target_data = extract_pdf_full_logic(f_new.read())
            if target_data:
                st.image(target_data['img'], use_container_width=True)

    if sel_name != "-- Chọn mẫu --" and f_new and 'target_data' in locals():
        if st.button("BẮT ĐẦU SO SÁNH", use_container_width=True):
            render_comparison_logic(target_data, repo_dict[sel_name])
