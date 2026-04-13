import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH GIAO DIỆN =================
st.set_page_config(layout="wide", page_title="AI V20.0 - Fashion Auditor", page_icon="🛡️")

# CSS giả lập giao diện chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTable { font-size: 11px !important; }
    thead th { background-color: #f0f2f6 !important; }
    .css-1offfwp { background-color: #262730 !important; } /* Sidebar dark */
    </style>
    """, unsafe_allow_html=True)

# Kết nối Supabase (Thay thông tin của bạn)
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

# ================= 2. HÀM XỬ LÝ DỮ LIỆU =================
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

def extract_pdf(file):
    specs, img_bytes = {}, None
    try:
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        doc.close()
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                for tb in page.extract_tables():
                    df = pd.DataFrame(tb)
                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row if c]
                        if any(x in " ".join(row_up) for x in ["POM", "DESCRIPTION"]):
                            n_idx, v_idx = 0, 1
                            for i, v in enumerate(row_up):
                                if "DESC" in v or "POM" in v: n_idx = i
                                if any(x in v for x in ["NEW", "SAMPLE", "M", "32"]): v_idx = i
                            for d_idx in range(r_idx + 1, len(df)):
                                name = str(df.iloc[d_idx, n_idx]).upper()
                                val = parse_val(df.iloc[d_idx, v_idx])
                                if len(name) > 3 and val > 0: specs[name] = val
                            break
        return {"specs": specs, "img": img_bytes}
    except: return None

# ================= 3. SIDEBAR GIAO DIỆN MẪU =================
with st.sidebar:
    st.markdown("### 🛡️ AI V20.0")
    st.button("📁 Kho mẫu: 6 file", use_container_width=True)
    st.button("🧩 CẬP NHẬT KHO MẪU", use_container_width=True)
    st.divider()
    st.write("**Size cần so sánh**")
    size = st.selectbox("Chọn size", ["XS", "S", "M", "L", "XL"], index=2)
    st.write("**CHỌN MẪU THỦ CÔNG**")
    mode = st.selectbox("Chế độ", ["Tự động tìm", "Chọn từ danh sách"])

# ================= 4. HIỂN THỊ CHI TIẾT & XUẤT EXCEL =================
st.subheader("🔍 ĐỐI SOÁT THÔNG SỐ SẢN PHẨM")
uploaded_file = st.file_uploader("Upload Techpack", type="pdf", label_visibility="collapsed")

if uploaded_file:
    target = extract_pdf(uploaded_file)
    if target and target["specs"]:
        db_res = supabase.table("ai_data").select("*").execute()
        if db_res.data:
            # So khớp AI
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().detach().cpu().numpy().reshape(1, -1)
            
            matches = []
            for item in db_res.data:
                v_ref = np.atleast_2d(item["vector"]).astype(np.float32)
                score = float(cosine_similarity(v_test, v_ref))
                matches.append({"item": item, "score": score})
            
            best = sorted(matches, key=lambda x: x['score'], reverse=True)[0]
            
            # CHIA 2 CỘT SONG SONG
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("📄 BẢN ĐANG KIỂM")
                st.image(target["img"])
                df_l = pd.DataFrame([{"STT": i+1, "Hạng mục": k, "Số đo": v} for i, (k,v) in enumerate(target["specs"].items())])
                st.table(df_l)

            with col2:
                st.caption(f"✨ MẪU GỐC (Khớp {best['score']*100:.1f}%)")
                st.image(best['item']['image_url'])
                
                # Tính toán bảng so sánh
                ref_specs = best['item']['spec_json']
                data_right = []
                for k, v in target["specs"].items():
                    k_clean = re.sub(r'[^A-Z0-9]', '', k)
                    v_ref = 0
                    for rk, rv in ref_specs.items():
                        if re.sub(r'[^A-Z0-9]', '', rk) == k_clean:
                            v_ref = rv; break
                    
                    diff = round(v - v_ref, 3)
                    status = "Khớp" if abs(diff) < 0.125 else "Lệch"
                    data_right.append({"Thông số": k, "Mới": v, "Kho mẫu": v_ref, "Chênh lệch": status})
                
                df_r = pd.DataFrame(data_right)
                
                # Hàm tô màu kết quả
                def color_status(val):
                    color = 'green' if val == 'Khớp' else 'red'
                    return f'color: {color}; font-weight: bold'
                
                st.table(df_r.style.applymap(color_status, subset=['Chênh lệch']))

            # ================= 5. LOGIC XUẤT EXCEL =================
            st.divider()
            
            # Tạo buffer để lưu file Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_r.to_excel(writer, index=False, sheet_name='Audit_Report')
                
                workbook  = writer.book
                worksheet = writer.sheets['Audit_Report']
                
                # Định dạng Header
                header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
                for col_num, value in enumerate(df_r.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Định dạng màu xanh/đỏ cho cột Chênh lệch
                format_green = workbook.add_format({'font_color': 'green', 'bold': True})
                format_red = workbook.add_format({'font_color': 'red', 'bold': True})
                
                worksheet.conditional_format('D2:D100', {
                    'type':     'cell',
                    'criteria': '==',
                    'value':    '"Khớp"',
                    'format':   format_green
                })
                worksheet.conditional_format('D2:D100', {
                    'type':     'cell',
                    'criteria': '==',
                    'value':    '"Lệch"',
                    'format':   format_red
                })

            st.download_button(
                label="📥 TẢI BÁO CÁO EXCEL",
                data=output.getvalue(),
                file_name=f"Audit_{best['item']['file_name']}.xlsx",
                mime="application/vnd.ms-excel",
                type="primary"
            )
