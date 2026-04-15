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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor Gold", page_icon="👖")

# ================= 2. HÀM AI & XỬ LÝ (GIỮ NGUYÊN LOGIC CŨ) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def parse_val(t):
    try:
        if t is None or str(t).strip() == "": return 0
        txt = str(t).replace(',', '.').replace('"', '').strip().lower()
        txt = re.sub(r'(cm|inch|in|mm|yds)$', '', txt)
        match = re.findall(r'(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v_str = match[0]
        if ' ' in v_str:
            p = v_str.split(); return float(p[0]) + eval(p[1])
        elif '/' in v_str: return eval(v_str)
        else: return float(v_str)
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad(): return model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy().astype(float).tolist()

def extract_pdf_multi_size(file):
    # [Giữ nguyên hàm extract_pdf_multi_size cũ của bạn ở đây]
    # Lưu ý: Đảm bảo hàm này return đúng dict như code cũ
    pass 

# ================= 3. SIDEBAR: QUẢN LÝ KHO =================
with st.sidebar:
    st.header("🏢 QUẢN LÝ KHO")
    
    # Hiển thị số lượng mẫu hiện có
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    total_samples = res_count.count if res_count.count else 0
    st.metric("Tổng số mẫu trong kho", total_samples)
    
    st.divider()
    new_files = st.file_uploader("Nạp mẫu mới (PDF)", accept_multiple_files=True)
    if new_files and st.button("NẠP VÀO HỆ THỐNG"):
        for f in new_files:
            data = extract_pdf_multi_size(f)
            if data:
                path = f"lib_{re.sub(r'\W+', '', f.name)}.png"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                supabase.table("ai_data").insert({
                    "file_name": f.name, "vector": get_image_vector(data['img']),
                    "spec_json": data['all_specs'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "customer_name": data['customer']
                }).execute()
        st.success("Đã cập nhật kho mẫu!"); st.rerun()

# ================= 4. GIAO DIỆN CHÍNH & ĐỐI SOÁT =================
st.title("🔍 AI SMART AUDITOR - V96 PRO")
file_audit = st.file_uploader("📤 Tải file PDF cần kiểm tra (Audit)", type="pdf")

if file_audit:
    target = extract_pdf_multi_size(file_audit)
    if target and target["all_specs"]:
        # Tìm kiếm mẫu tương đồng nhất
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            df_db = pd.DataFrame(res.data)
            t_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)
            df_db['sim'] = cosine_similarity(t_vec, np.array([v for v in df_db['vector']])).flatten()
            best = df_db.sort_values('sim', ascending=False).iloc[0]
            
            # Hiển thị kết quả tìm kiếm
            col1, col2 = st.columns(2)
            with col1: st.image(target['img'], caption="Ảnh file Audit", use_container_width=True)
            with col2: st.image(best['image_url'], caption=f"Mẫu khớp nhất: {best['file_name']} (Sim: {best['sim']:.2%})", use_container_width=True)
            
            # Đối soát chi tiết
            st.subheader("📊 Bảng so sánh thông số chi tiết")
            sel_size = st.selectbox("Chọn Size đối soát:", list(target['all_specs'].keys()))
            
            audit_specs = target['all_specs'].get(sel_size, {})
            db_specs_all = best['spec_json']
            # Tìm size tương ứng trong DB (nếu không khớp tên size thì lấy size đầu tiên)
            ref_size = sel_size if sel_size in db_specs_all else list(db_specs_all.keys())[0]
            ref_specs = db_specs_all.get(ref_size, {})

            # Tạo bảng so sánh
            comparison_data = []
            for pom, val_audit in audit_specs.items():
                val_ref = ref_specs.get(pom, 0)
                diff = round(val_audit - val_ref, 4)
                status = "✅ Đạt" if abs(diff) < 0.1 else "❌ Lệch" # Ngưỡng 0.1 có thể chỉnh
                
                comparison_data.append({
                    "Thông số (POM)": pom,
                    f"File Audit ({sel_size})": val_audit,
                    f"Mẫu Kho ({ref_size})": val_ref,
                    "Chênh lệch": diff,
                    "Trạng thái": status
                })
            
            df_compare = pd.DataFrame(comparison_data)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

            # NÚT XUẤT EXCEL
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_compare.to_excel(writer, index=False, sheet_name='Audit_Report')
            
            st.download_button(
                label="📥 XUẤT BÁO CÁO EXCEL",
                data=output.getvalue(),
                file_name=f"Audit_{best['file_name']}_{sel_size}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("Không tìm thấy bảng thông số trong file PDF này.")
