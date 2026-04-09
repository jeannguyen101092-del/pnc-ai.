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

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V34.8", page_icon="📊")

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

# ================= TOOLS =================
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
                tbs = page.extract_tables()
                for tb in tbs:
                    df = pd.DataFrame(tb).dropna(how='all')
                    for idx, row in df.iterrows():
                        row_up = [str(x).upper() for x in row if x]
                        # Nhận diện bảng thông số qua từ khóa Desc hoặc Point of Measure
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

# ================= LOAD DATA =================
samples = []
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res.data else []
except: pass

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 Kho dữ liệu gốc")
    st.metric("Tổng mẫu", len(samples))
    files = st.file_uploader("Nạp Techpack (PDF)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("🚀 Bắt đầu nạp"):
        for f in files:
            d = extract_techpack(f)
            if d and d['img']:
                name = f.name.replace(".pdf","")
                try:
                    supabase.storage.from_(BUCKET).upload(f"{name}.png", d['img'], {"content-type":"image/png", "x-upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(f"{name}.png")
                    vec = get_vector(d['img'])
                    # Lưu bảng dưới dạng list of dicts
                    specs = json.dumps([df.to_dict(orient='records') for df in d['tables']])
                    supabase.table("ai_data").upsert({"file_name":name, "vector":vec, "spec_json":specs, "image_url":img_url}).execute()
                except: pass
        st.success("Nạp xong!"); st.rerun()

# ================= MAIN =================
st.title("🔍 AI FASHION AUDITOR V34.8")
test_file = st.file_uploader("Kéo tệp PDF cần kiểm tra vào đây", type=["pdf"])

if test_file:
    test_data = extract_techpack(test_file)
    if test_data and test_data['img']:
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(test_data['img'], caption="Mẫu đang kiểm tra", use_container_width=True)
            t_vec = get_vector(test_data['img'])

        if samples and t_vec:
            # AI gợi ý mẫu
            results = [{"data": s, "sim": cosine_similarity([t_vec], [s['vector']])} for s in samples]
            best_ai = max(results, key=lambda x: x['sim'])
            
            with col2:
                st.subheader("⚙️ Thiết lập đối soát")
                # Chọn mã hàng thủ công
                s_names = [s['file_name'] for s in samples]
                selected_name = st.selectbox("📌 Chọn mã hàng đối soát:", s_names, index=s_names.index(best_ai['data']['file_name']))
                
                ref_data = next(s for s in samples if s['file_name'] == selected_name)
                sim_val = round(cosine_similarity([t_vec], [ref_data['vector']])[0][0] * 100, 1)
                st.write(f"Độ tương đồng AI: **{sim_val}%**")
                st.progress(sim_val/100)

                # --- XỬ LÝ CHỌN SIZE TRONG BẢNG ---
                if test_data['tables'] and ref_data['spec_json']:
                    # Lấy bảng có nhiều cột nhất (thường là bảng thông số)
                    df_test = max(test_data['tables'], key=lambda x: len(x.columns))
                    
                    # Danh sách từ khóa gây nhiễu cần loại bỏ khỏi danh sách Size
                    noise =
                    
                    # Lọc lấy các cột thực sự là Size
                    actual_sizes = [c for c in df_test.columns if c and not any(n in str(c).upper() for n in noise)]
                    # Ưu tiên các cột có tên ngắn (Size thường ngắn)
                    actual_sizes = [c for c in actual_sizes if len(str(c)) < 10]

                    c_s1, c_s2 = st.columns(2)
                    with c_s1: sel_size = st.selectbox("🎯 Chọn Size trong bảng thông số:", actual_sizes)
                    with c_s2: st.info(f"Đối soát: {selected_name}")

                    # --- SO SÁNH TỰ ĐỘNG ---
                    if sel_size:
                        audit_data = []
                        # Load bảng gốc từ DB
                        try:
                            ref_tables = json.loads(ref_data['spec_json'])
                            df_ref = pd.DataFrame(ref_tables[0]) if ref_tables else pd.DataFrame()
                        except: df_ref = pd.DataFrame()

                        for _, row_t in df_test.iterrows():
                            # Tìm cột mô tả
                            d_col = next((c for c in df_test.columns if any(k in str(c).upper() for k in ['DESC', 'ITEM', 'POINT'])), None)
                            if not d_col: continue
                            
                            desc = str(row_t[d_col]).strip().upper()
                            if not desc or desc == 'NAN' or len(desc) < 3: continue
                            
                            # Khớp dòng tự động
                            if not df_ref.empty:
                                ref_d_col = df_ref.columns[0] # Giả định cột 0 là Desc
                                match = df_ref[df_ref[ref_d_col].astype(str).str.upper().str.contains(desc, na=False, regex=False)]
                                
                                if not match.empty:
                                    try:
                                        v1 = float(re.findall(r"\d+\.?\d*", str(row_t.get(sel_size, '0')))[0])
                                        v2 = float(re.findall(r"\d+\.?\d*", str(match.iloc[0].get(sel_size, '0')))[0])
                                        diff = round(v1 - v2, 2)
                                        audit_data.append({
                                            "Hạng mục": desc, 
                                            f"Kiểm ({sel_size})": v1, 
                                            f"Gốc ({sel_size})": v2, 
                                            "Lệch": diff, 
                                            "Kết quả": "✅ OK" if abs(diff) <= 0.5 else "❌ LỆCH"
                                        })
                                    except: pass
                        
                        if audit_data:
                            res_df = pd.DataFrame(audit_data)
                            st.table(res_df)
                            st.download_button("📥 Xuất File Excel", to_excel(res_df), f"Audit_{selected_name}.xlsx")
                        else:
                            st.warning("⚠️ Không khớp được hạng mục nào. Kiểm tra lại cột Description.")
    else:
        st.error("Không đọc được dữ liệu PDF.")

st.markdown("---")
st.caption("AI Fashion Auditor V34.8 | Fix: Smart Size Selector & Auto-Matching")
