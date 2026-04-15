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

st.set_page_config(layout="wide", page_title="AI Universal Auditor V114", page_icon="🏢")

if 'up_key' not in st.session_state:
    st.session_state.up_key = 0

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

model_ai = load_model()

# ================= UI =================
st.markdown("""
<style>
.stMetric { background-color: #ffffff; padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; }
.status-khop { color: #28a745; font-weight: bold; }
.status-lech { color: #dc3545; font-weight: bold; }
thead th { background-color: #f8f9fa !important; color: #333 !important; }
</style>
""", unsafe_allow_html=True)

# ================= CATEGORY =================
def smart_detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    t = re.sub(r"[^A-Z0-9 ]", " ", t)

    if any(x in t for x in ["INSEAM","THIGH","LEG OPENING"]):
        return "QUẦN"
    if any(x in t for x in ["CHEST","BUST","SLEEVE"]):
        return "ÁO"

    if "DRESS" in t or "SKIRT" in t:
        return "VÁY"

    return "KHÁC"

# ================= PARSE VALUE =================
def parse_val_universal(t):
    try:
        if not t: return 0
        txt = str(t).replace(',', '.')
        match = re.search(r'(\d+\.\d+|\d+)', txt)
        return float(match.group(1)) if match else 0
    except:
        return 0

# ================= IMAGE VECTOR =================
def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return vec.tolist()

# ================= EXTRACT =================
def extract_pdf(file):
    all_specs, img_bytes = {}, None
    try:
        file.seek(0)
        pdf_content = file.read()

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap().tobytes("png")

        full_text = " ".join([p.get_text() for p in doc])
        doc.close()

        category = smart_detect_category(full_text, file.name)

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    
                    n_col = -1
                    size_cols = {}

                    for r_idx, row in df.iterrows():
                        row_up = [str(c).upper() for c in row]

                        for i,v in enumerate(row_up):
                            if "POM" in v or "DESCRIPTION" in v:
                                n_col = i

                        for i,v in enumerate(row_up):
                            if i != n_col and len(v)<6:
                                size_cols[i] = v

                        if n_col != -1 and size_cols:
                            break

                    if n_col == -1:
                        continue

                    for s_col, s_name in size_cols.items():
                        all_specs.setdefault(s_name, {})
                        for r in range(len(df)):
                            pom = str(df.iloc[r, n_col]).strip().upper()
                            val = parse_val_universal(df.iloc[r, s_col])
                            if len(pom)>2 and val>0:
                                all_specs[s_name][pom] = val

                if all_specs:
                    break

        return {
            "all_specs": all_specs,
            "img": img_bytes,
            "category": category
        }

    except Exception as e:
        print("ERROR:", e)
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO")
    cust_input = st.text_input("Tên khách hàng", value="EXPRESS")
    files = st.file_uploader("Upload Techpack", accept_multiple_files=True)

    if files and st.button("UPLOAD"):
        for f in files:
            data = extract_pdf(f)
            if data and data['all_specs']:
                vec = get_image_vector(data['img'])

                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": data['all_specs'],
                    "category": data['category'],
                    "customer_name": cust_input
                }).execute()

        st.success("DONE")

# ================= MAIN =================
st.title("🔍 AI UNIVERSAL AUDITOR")

file_audit = st.file_uploader("Upload file audit", type="pdf")

if file_audit:
    data = extract_pdf(file_audit)

    if data and data['all_specs']:
        st.success("Đã đọc file")

        res = supabase.table("ai_data").select("*").execute()

        if res.data:
            df = pd.DataFrame(res.data)

            target_vec = np.array(get_image_vector(data['img'])).reshape(1,-1)
            db_vecs = np.array(df['vector'].tolist())

            df['sim'] = cosine_similarity(target_vec, db_vecs)[0]

            top = df.sort_values("sim", ascending=False).head(3)

            for _, row in top.iterrows():
                st.subheader(row['file_name'])

                lib = row['spec_json']
                audit = data['all_specs']

                size = list(audit.keys())[0]
                lib_size = list(lib.keys())[0]

                res_list = []
                for k,v in audit[size].items():
                    v2 = lib[lib_size].get(k,0)
                    diff = v - v2
                    res_list.append([k,v,v2,diff])

                st.table(pd.DataFrame(res_list, columns=["POM","Audit","Lib","Diff"]))
