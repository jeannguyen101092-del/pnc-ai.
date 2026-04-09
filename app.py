import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.10", page_icon="📊")

# ================= KẾT NỐI =================
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

# ================= CÔNG CỤ =================
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
    data = {"img": None, "tables": []}
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) > 0:
            data["img"] = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tbs = page.extract_tables() or []
                for tb in tbs:
                    df = pd.DataFrame(tb).dropna(how='all')
                    for idx, row in df.iterrows():
                        row_up = [str(x).upper() for x in row if x]
                        if any(re.search(r"(DESC|ITEM|POINT|MEASURE|POM)", s) for s in row_up):
                            df.columns = [str(c).replace('\n', ' ').strip().upper() for c in row]
                            df = df.iloc[idx+1:].reset_index(drop=True)
                            data["tables"].append(df)
                            break
        return data
    except: return None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Audit')
    return output.getvalue()

# ================= DỮ LIỆU =================
samples = []
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res.data else []
except: pass

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 Kho dữ liệu")
    files = st.file_uploader("Nạp mẫu gốc (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Bắt đầu nạp"):
        for f in files:
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                try:
                    supabase.storage.from_(BUCKET).upload(path=f"{name}.png", file=d['img'], file_options={"content-type":"image/png", "x-upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    vec = get_vector(d['img'])
                    specs = json.dumps([df.to_dict(orient='records') for df in d['tables']])
                    supabase.table("ai_data").upsert({"file_name":name, "vector":vec, "spec_json":specs, "image_url":img_url}).execute()
                except: pass
        st.success("Xong!"); st.rerun()

# ================= GIAO DIỆN CHÍNH =================
st.title("🔍 AI FASHION AUDITOR V34.10")
test_file = st.file_uploader("Kéo tệp PDF kiểm tra vào đây", type=["pdf"])

if test_file:
    test_data = extract_techpack(test_file)
    if test_data and test_data['img']:
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(test_data['img'], caption="Bản vẽ mẫu kiểm", use_container_width=True)
            t_vec = get_vector(test_data['img'])

        if samples and t_vec:
            # AI tìm mẫu
            results = []
            for s in samples:
                sim = cosine_similarity([t_vec], [s['vector']])[0][0]
                results.append({"data": s, "sim": sim})
            
            best_ai = max(results, key=lambda x: x['sim'])
            
            with col2:
                st.subheader("⚙️ Đối soát hệ thống")
                s_names = [s['file_name'] for s in samples]
                selected_name = st.selectbox("📌 Chọn mã hàng đối soát:", s_names, index=s_names.index(best_ai['data']['file_name']))
                
                ref_data = next(s for s in samples if s['file_name'] == selected_name)
                sim_val = round(cosine_similarity([t_vec], [ref_data['vector']])[0][0] * 100, 1)
                
                st.write(f"Độ tương đồng AI: **{sim_val}%**")
                st.progress(sim_val/100)

                if test_data['tables'] and ref_data['spec_json']:
                    df_test = max(test_data['tables'], key=lambda x: len(x.columns))
                    
                    # Dòng 132 (Đã sửa lỗi viết hoa/thường và đóng ngoặc)
                    noise =
                    
                    actual_sizes = [c for c in df_test.columns if c and not any(n in str(c).upper() for n in noise)]
                    actual_sizes = [c for c in actual_sizes if len(str(c)) < 10]

                    if actual_sizes:
                        sel_size = st.selectbox("🎯 Chọn Size từ bảng thông số:", actual_sizes)
                        
                        audit_list = []
                        try:
                            # Parse JSON chuỗi sang List rồi sang DataFrame
                            raw_spec = json.loads(ref_data['spec_json'])
                            df_ref = pd.DataFrame(raw_spec[0]) if isinstance(raw_spec, list) else pd.DataFrame()
                        except: df_ref = pd.DataFrame()

                        if not df_ref.empty:
                            for _, row_t in df_test.iterrows():
                                # Tìm cột Description
                                d_col = next((c for c in df_test.columns if any(k in str(c).upper() for k in ['DESC', 'ITEM', 'POINT'])), df_test.columns[0])
                                desc = str(row_t[d_col]).strip().upper()
                                if not desc or desc == 'NAN' or len(desc) < 3: continue
                                
                                # Khớp dòng tự động giữa 2 bảng
                                match = df_ref[df_ref.iloc[:, 0].astype(str).str.upper().str.contains(desc, na=False, regex=False)]
                                
                                if not match.empty:
                                    try:
                                        v1 = float(re.findall(r"\d+\.?\d*", str(row_t.get(sel_size, '0')))[0])
                                        v2 = float(re.findall(r"\d+\.?\d*", str(match.iloc[0].get(sel_size, '0')))[0])
                                        diff = round(v1 - v2, 2)
                                        audit_list.append({"Hạng mục": desc, f"Kiểm ({sel_size})": v1, f"Gốc ({sel_size})": v2, "Lệch": diff, "Kquả": "✅ OK" if abs(diff) <= 0.5 else "❌ LỆCH"})
                                    except: pass
                            
                            if audit_list:
                                res_df = pd.DataFrame(audit_list)
                                st.table(res_df)
                                st.download_button("📥 Xuất báo cáo Excel", to_excel(res_df), f"Report_{selected_name}.xlsx")
                            else:
                                st.warning("⚠️ Không tìm thấy hạng mục khớp nhau giữa 2 bảng.")
                    else:
                        st.warning("⚠️ Không tìm thấy cột Size hợp lệ trong bảng.")
    else:
        st.error("Không trích xuất được dữ liệu PDF.")
