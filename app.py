import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG (Thay URL/KEY của bạn) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.5", page_icon="📊")

# ================= KẾT NỐI HỆ THỐNG =================
@st.cache_resource
def init_supabase():
    try: return create_client(URL, KEY)
    except: return None

supabase = init_supabase()

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= CÔNG CỤ XỬ LÝ =================
def get_vector(img_bytes):
    try:
        if not img_bytes: return None
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
            return vec.tolist()
    except: return None

def extract_techpack(pdf_file):
    """Trích xuất ảnh và các bảng thông số kỹ thuật"""
    data = {"img": None, "tables": []}
    try:
        pdf_bytes = pdf_file.read()
        # 1. Trích xuất ảnh mẫu (Trang 1)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) > 0:
            data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        # 2. Trích xuất bảng dữ liệu
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tbs = page.extract_tables()
                if not tbs: continue
                for tb in tbs:
                    df = pd.DataFrame(tb)
                    # Tìm dòng Header chứa chữ 'DESCRIPTION'
                    for idx, row in df.iterrows():
                        row_up = [str(x).upper() for x in row if x]
                        if any("DESCRIPTION" in s for s in row_up):
                            df.columns = [str(c).strip().upper() for c in row]
                            df = df.iloc[idx+1:].reset_index(drop=True)
                            data["tables"].append(df)
                            break
        return data
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

def to_excel(df):
    """Tạo file Excel report"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Audit_Report')
        workbook = writer.book
        worksheet = writer.sheets['Audit_Report']
        # Format
        fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, fmt_header)
        worksheet.set_column('A:A', 35)
    return output.getvalue()

# ================= GIAO DIỆN SIDEBAR =================
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res.data else []
except:
    samples = []

with st.sidebar:
    st.header("📂 Kho dữ liệu gốc")
    st.metric("Tổng số mẫu", len(samples))
    
    files = st.file_uploader("Nạp Techpack mẫu (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if files and st.button("🚀 Bắt đầu nạp"):
        progress = st.progress(0)
        for i, f in enumerate(files):
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                img_path = f"{name}.png"
                try:
                    # Upload ảnh lên Storage
                    supabase.storage.from_(BUCKET).upload(
                        path=img_path, 
                        file=d['img'], 
                        file_options={"content-type": "image/png", "x-upsert": "true"}
                    )
                    img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
                    
                    # Tạo vector AI
                    vec = get_vector(d['img'])
                    
                    # Chuyển đổi list các DataFrame thành chuỗi JSON
                    list_of_dicts = [df.to_dict(orient='records') for df in d['tables']]
                    specs_json_str = json.dumps(list_of_dicts)
                    
                    # Lưu vào Database
                    supabase.table("ai_data").upsert({
                        "file_name": name,
                        "vector": vec,
                        "spec_json": specs_json_str,
                        "image_url": img_url
                    }).execute()
                    st.success(f"Đã nạp: {name}")
                except Exception as e:
                    st.error(f"Lỗi nạp {name}: {e}")
            progress.progress((i + 1) / len(files))
        st.rerun()

# ================= GIAO DIỆN CHÍNH (AUDIT) =================
st.title("🔍 AI FASHION AUDITOR V34.5")
test_file = st.file_uploader("Kéo tệp PDF cần kiểm tra vào đây", type=["pdf"])

if test_file:
    test_data = extract_techpack(test_file)
    if test_data and test_data['img']:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(test_data['img'], caption="Ảnh mẫu đang kiểm tra", use_container_width=True)
            t_vec = get_vector(test_data['img'])
            
        if samples and t_vec:
            # So sánh AI
            results = []
            for s in samples:
                sim = cosine_similarity([t_vec], [s['vector']])[0][0]
                results.append({"data": s, "sim": sim})
            
            best = max(results, key=lambda x: x['sim'])
            
            with col2:
                sim_score = round(best['sim'] * 100, 1)
                st.subheader(f"Độ tương đồng AI: {sim_score}%")
                st.progress(float(best['sim']))
                st.info(f"Khớp nhất với mẫu: **{best['data']['file_name']}**")

                # Xử lý so sánh thông số
                if test_data['tables'] and best['data']['spec_json']:
                    # Lấy bảng đầu tiên của file kiểm tra
                    df_test = test_data['tables'][0]
                    # Load dữ liệu gốc từ JSON
                    ref_list = json.loads(best['data']['spec_json'])
                    df_ref = pd.DataFrame(ref_list[0]) if ref_list else pd.DataFrame()

                    # Lọc lấy các cột Size (bỏ Description, No...)
                    ignore = ['DESCRIPTION', 'NO', 'TOL', 'TOLERANCE', 'INDEX', 'STT', '']
                    size_cols = [c for c in df_test.columns if c and not any(x in str(c).upper() for x in ignore)]
                    
                    sel_size = st.selectbox("🎯 Chọn Size để đối soát:", size_cols)

                    if sel_size:
                        audit_results = []
                        for _, row_t in df_test.iterrows():
                            desc = str(row_t.get('DESCRIPTION', '')).strip().upper()
                            if not desc or desc == 'NAN': continue
                            
                            # Tìm dòng tương ứng ở mẫu gốc
                            match = df_ref[df_ref['DESCRIPTION'].astype(str).str.upper() == desc]
                            if not match.empty:
                                try:
                                    v_test = float(re.findall(r"\d+\.?\d*", str(row_t.get(sel_size, '0')))[0])
                                    v_ref = float(re.findall(r"\d+\.?\d*", str(match.iloc[0].get(sel_size, '0')))[0])
                                    diff = round(v_test - v_ref, 2)
                                    audit_results.append({
                                        "Mô tả (Description)": desc,
                                        "Thông số kiểm": v_test,
                                        "Thông số gốc": v_ref,
                                        "Chênh lệch": diff,
                                        "Kết quả": "✅ Khớp" if abs(diff) <= 0.5 else "❌ Lệch"
                                    })
                                except: pass
                        
                        if audit_results:
                            final_df = pd.DataFrame(audit_results)
                            st.table(final_df)
                            
                            # Nút Xuất Excel
                            xlsx = to_excel(final_df)
                            st.download_button(
                                label="📥 Tải Báo Cáo Excel",
                                data=xlsx,
                                file_name=f"Audit_{best['data']['file_name']}_{sel_size}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
    else:
        st.warning("Không tìm thấy dữ liệu bảng hoặc ảnh trong file PDF này.")

st.markdown("---")
st.caption("AI Fashion Auditor v34.5 | Database: Supabase | Engine: ResNet50")
