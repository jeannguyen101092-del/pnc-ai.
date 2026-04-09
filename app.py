import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.4", page_icon="📊")

# ================= INIT =================
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

# ================= CORE FUNCTIONS =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        with torch.no_grad():
            return model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except: return None

def extract_techpack(pdf_file):
    full_data = {"img": None, "tables": []}
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) > 0:
            full_data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for tb in tables:
                    if len(tb) < 2: continue
                    df = pd.DataFrame(tb)
                    for idx, row in df.iterrows():
                        row_str = [str(x).upper() for x in row if x]
                        if any("DESCRIPTION" in s for s in row_str):
                            df.columns = [str(c).strip().upper() for c in row]
                            df = df.iloc[idx+1:].reset_index(drop=True)
                            full_data["tables"].append(df)
                            break
        return full_data
    except: return None

# Hàm chuyển đổi DF sang file Excel để tải về
def to_excel(df, size_name, match_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Audit_Report')
        workbook = writer.book
        worksheet = writer.sheets['Audit_Report']
        
        # Định dạng header
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        worksheet.set_column('A:A', 40) # Cột Description rộng hơn
        output.getbuffer()
    return output.getvalue()

# ================= SIDEBAR & MAIN =================
samples = []
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res.data else []
except: pass

with st.sidebar:
    st.header("📦 Kho dữ liệu")
    files = st.file_uploader("Nạp mẫu gốc (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Nạp dữ liệu"):
        for f in files:
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                img_path = f"{name}.png"
                try:
                    supabase.storage.from_(BUCKET).upload(path=img_path, file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
                    vec = get_vector(d['img'])
                    specs_json = d['tables'][0].to_json() if d['tables'] else "{}"
                    supabase.table("ai_data").upsert({"file_name": name, "vector": vec, "spec_json": specs_json, "image_url": img_url}).execute()
                except: pass
        st.success("Đã nạp xong!"); st.rerun()

st.title("🔍 AI FASHION AUDITOR V34.4")
test_file = st.file_uploader("Kéo tệp PDF cần kiểm tra vào đây", type=["pdf"])

if test_file:
    test_data = extract_techpack(test_file)
    if test_data and test_data['img']:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(test_data['img'], caption="Ảnh mẫu kiểm", use_container_width=True)
            t_vec = get_vector(test_data['img'])

        if samples and t_vec:
            results = [{"data": s, "sim": cosine_similarity([t_vec], [s['vector']])[0][0]} for s in samples]
            best = max(results, key=lambda x: x['sim'])
            
            with col2:
                st.subheader(f"Độ tương đồng AI: {round(best['sim']*100, 1)}%")
                st.progress(float(best['sim']))
                st.info(f"Khớp nhất với: **{best['data']['file_name']}**")

                if test_data['tables'] and best['data']['spec_json']:
                    df_test = test_data['tables'][0]
                    df_ref = pd.read_json(io.StringIO(best['data']['spec_json']))

                    ignore = ['DESCRIPTION', 'NO', 'TOL', 'TOLERANCE', 'NONE', 'INDEX', 'STT']
                    all_cols = [c for c in df_test.columns if c and not any(ig in str(c).upper() for ig in ignore)]
                    
                    selected_size = st.selectbox("🎯 Chọn Size để đối soát:", all_cols)

                    if selected_size:
                        comp_list = []
                        for _, row_t in df_test.iterrows():
                            desc_t = str(row_t.get('DESCRIPTION', '')).strip().upper()
                            if not desc_t or desc_t == 'NAN': continue
                            
                            match_ref = df_ref[df_ref['DESCRIPTION'].astype(str).str.upper() == desc_t]
                            
                            if not match_ref.empty:
                                try:
                                    v1 = float(re.findall(r"\d+\.?\d*", str(row_t.get(selected_size, '0')))[0])
                                    v2 = float(re.findall(r"\d+\.?\d*", str(match_ref.iloc[0].get(selected_size, '0')))[0])
                                    diff = round(v1 - v2, 2)
                                    comp_list.append({
                                        "Hạng mục": desc_t,
                                        f"Mẫu Kiểm ({selected_size})": v1,
                                        f"Mẫu Gốc ({selected_size})": v2,
                                        "Chênh lệch": diff,
                                        "Kết quả": "OK" if abs(diff) <= 0.5 else "LỆCH"
                                    })
                                except: pass
                        
                        if comp_list:
                            final_df = pd.DataFrame(comp_list)
                            st.table(final_df)
                            
                            # NÚT XUẤT EXCEL
                            excel_data = to_excel(final_df, selected_size, best['data']['file_name'])
                            st.download_button(
                                label="📥 Tải Báo Cáo Đối Soát (Excel)",
                                data=excel_data,
                                file_name=f"Audit_{best['data']['file_name']}_{selected_size}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
