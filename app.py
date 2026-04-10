# ==============================
# AI FASHION AUDITOR V45 - ENTERPRISE VERSION
# FULL SYSTEM: PDF PARSER + AI MATCHING + DB + UI + QA TOOL
# ==============================

import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, time, base64
import torch, json, math
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45", page_icon="🔥")

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

model_ai = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= TEXT CLEAN =================
def clean_text(t):
    t = str(t).upper()
    t = re.sub(r'[^A-Z0-9 ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

# ================= VALUE PARSE =================
def parse_val(x):
    try:
        x = str(x).replace(',', '.').strip()
        if not x or x in ['-', 'N/A']: return None
        if ' ' in x and '/' in x:
            a,b = x.split()
            return float(a) + eval(b)
        if '/' in x:
            return eval(x)
        return float(re.findall(r"\d+\.\d+|\d+", x)[0])
    except:
        return None

# ================= PDF EXTRACT =================
def extract_pdf(file):
    specs = {}
    image_bytes = None

    pdf_bytes = file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # IMAGE HIGH RES
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(3,3))
    image_bytes = pix.tobytes("png")

    # TABLE PARSE
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for tb in tables:
                df = pd.DataFrame(tb).fillna("")

                header_row = -1
                pom_idx = -1
                size_cols = []

                for r_idx, row in df.head(15).iterrows():
                    row_up = [str(c).upper() for c in row]

                    for i, c in enumerate(row_up):
                        if "DESCRIPTION" in c:
                            pom_idx = i

                        if re.fullmatch(r'(XS|S|M|L|XL|XXL|\d+)', c):
                            size_cols.append(i)

                    if pom_idx != -1 and size_cols:
                        header_row = r_idx
                        break

                if header_row == -1:
                    continue

                val_idx = size_cols[len(size_cols)//2]

                for i in range(header_row+1, len(df)):
                    name = clean_text(df.iloc[i, pom_idx])
                    val = parse_val(df.iloc[i, val_idx])

                    if name and val:
                        specs[name] = val

    return specs, image_bytes

# ================= VECTOR =================
def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model_ai(t).flatten().numpy()
    return vec

# ================= STORAGE =================
def upload_image(file_name, img_bytes):
    path = f"techpack/{file_name}.png"
    supabase.storage.from_(BUCKET).upload(path, img_bytes, {"upsert":"true"})
    return supabase.storage.from_(BUCKET).get_public_url(path)["publicUrl"]

# ================= SAVE DB =================
def save_all(file_name, specs, img_bytes, vec):
    img_url = upload_image(file_name, img_bytes)

    supabase.table("ai_data").insert({
        "file_name": file_name,
        "image_url": img_url,
        "vector": vec.tolist()
    }).execute()

    batch = []
    for k,v in specs.items():
        batch.append({"file_name":file_name,"pom_name":k,"value":v})

    for i in range(0,len(batch),50):
        supabase.table("ai_specs").insert(batch[i:i+50]).execute()

# ================= MATCH =================
def find_best_match(vec):
    db = supabase.table("ai_data").select("*").execute()
    results = []

    for d in db.data:
        v = np.array(d['vector']).reshape(1,-1)
        sim = cosine_similarity(vec.reshape(1,-1), v)[0][0]*100
        results.append((d, sim))

    return sorted(results, key=lambda x: x[1], reverse=True)

# ================= COMPARE SPEC =================
def compare_specs(new_specs, ref_file):
    ref = supabase.table("ai_specs").select("*").eq("file_name", ref_file).execute()
    ref_dict = {i['pom_name']:i['value'] for i in ref.data}

    rows = []
    for k,v in new_specs.items():
        ref_v = ref_dict.get(k,0)
        diff = v - ref_v
        status = "OK"

        if abs(diff) > 0.5:
            status = "FAIL"

        rows.append([k,v,ref_v,diff,status])

    return pd.DataFrame(rows, columns=["POM","NEW","REF","DIFF","STATUS"])

# ================= UI =================
st.title("🔥 AI FASHION AUDITOR V45 - ENTERPRISE")

mode = st.sidebar.selectbox("Mode", ["Upload Library","Audit","Dashboard"])

# ================= UPLOAD =================
if mode == "Upload Library":
    files = st.file_uploader("Upload Techpack", accept_multiple_files=True)

    if files:
        for f in files:
            specs, img = extract_pdf(f)
            vec = get_vector(img)
            save_all(f.name, specs, img, vec)
            st.success(f"Saved {f.name}")

# ================= AUDIT =================
if mode == "Audit":
    f = st.file_uploader("Upload file check")

    if f:
        specs, img = extract_pdf(f)
        vec = get_vector(img)

        matches = find_best_match(vec)

        for m, sim in matches[:3]:
            st.subheader(f"{m['file_name']} - {sim:.2f}%")

            col1, col2 = st.columns(2)
            col1.image(img)
            col2.image(m['image_url'])

            df = compare_specs(specs, m['file_name'])

            st.dataframe(df.style.applymap(lambda x: 'background-color:red' if x=='FAIL' else ''))

            st.download_button("Download Excel", df.to_csv(index=False), file_name="audit.csv")

# ================= DASHBOARD =================
if mode == "Dashboard":
    data = supabase.table("ai_data").select("*").execute()

    st.metric("Total Styles", len(data.data))

    names = [d['file_name'] for d in data.data]
    st.selectbox("Select Style", names)

# ================= END =================
