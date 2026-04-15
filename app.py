import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI UNIVERSAL AUDITOR", page_icon="🏢")

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

model_ai = load_model()

# ================= IMAGE VECTOR =================
def get_vec(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        return model_ai(tf(img).unsqueeze(0)).flatten().numpy().tolist()

# ================= VALUE =================
def parse_val(t):
    try:
        txt = str(t).replace(",",".")
        m = re.search(r'(\d+\.\d+|\d+)', txt)
        return float(m.group(1)) if m else 0
    except:
        return 0

# ================= POM FILTER =================
def is_valid_pom(p):
    p = str(p).upper()

    blacklist = [
        "BUTTON","ZIPPER","LABEL","TRIM",
        "DRAWCORD","CORD","FABRIC","THREAD",
        "DESCRIPTION","STYLE","COLOR","WASH"
    ]

    if any(b in p for b in blacklist):
        return False

    valid = [
        "WAIST","HIP","THIGH","INSEAM","OUTSEAM",
        "CHEST","SLEEVE","LENGTH","SHOULDER","LEG"
    ]

    return any(v in p for v in valid)

# ================= NORMALIZE =================
def normalize_pom(p):
    p = str(p).upper()

    rules = {
        "WAIST":["WAIST"],
        "HIP":["HIP","SEAT"],
        "THIGH":["THIGH"],
        "INSEAM":["INSEAM"],
        "OUTSEAM":["OUTSEAM"],
        "LEG OPENING":["LEG"],
        "CHEST":["CHEST","BUST"],
        "SLEEVE":["SLEEVE"],
        "LENGTH":["LENGTH"]
    }

    for k,v in rules.items():
        if any(x in p for x in v):
            return k
    return p

# ================= TABLE SCORE =================
def score_table(df):
    text = " ".join(df.astype(str).values.flatten()).upper()
    keys = ["WAIST","HIP","THIGH","INSEAM","OUTSEAM","CHEST"]
    return sum(text.count(k) for k in keys)

# ================= EXTRACT =================
def extract_pdf(file):
    all_specs, img_bytes = {}, None

    file.seek(0)
    pdf_content = file.read()

    doc = fitz.open(stream=pdf_content, filetype="pdf")
    img_bytes = doc.load_page(0).get_pixmap().tobytes("png")
    doc.close()

    best_df = None
    best_score = 0

    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            for tb in page.extract_tables():
                df = pd.DataFrame(tb).fillna("")
                sc = score_table(df)
                if sc > best_score:
                    best_score = sc
                    best_df = df

    if best_df is None:
        return {"all_specs": {}, "img": img_bytes}

    df = best_df

    n_col = -1
    size_cols = {}

    for row in df.values:
        row_up = [str(c).upper() for c in row]

        for i,v in enumerate(row_up):
            if "POM" in v or "DESCRIPTION" in v:
                n_col = i

        for i,v in enumerate(row_up):
            if i != n_col and len(v) <= 5:
                size_cols[i] = v

        if n_col != -1 and size_cols:
            break

    for s_col, s_name in size_cols.items():
        all_specs.setdefault(s_name, {})

        for r in range(len(df)):
            pom = df.iloc[r, n_col]
            val = parse_val(df.iloc[r, s_col])

            if is_valid_pom(pom) and val > 0:
                pom = normalize_pom(pom)
                all_specs[s_name][pom] = val

    return {"all_specs": all_specs, "img": img_bytes}

# ================= MATCH =================
def match_pom(aud_keys, lib_keys):
    vec = TfidfVectorizer().fit(aud_keys + lib_keys)

    aud_vec = vec.transform(aud_keys).toarray()
    lib_vec = vec.transform(lib_keys).toarray()

    sim = cosine_similarity(aud_vec, lib_vec)

    mapping = {}
    for i, a in enumerate(aud_keys):
        idx = np.argmax(sim[i])
        score = sim[i][idx]

        if score > 0.5:
            mapping[a] = (lib_keys[idx], score)
        else:
            mapping[a] = (None, score)

    return mapping

# ================= UI =================
st.sidebar.header("📂 KHO")
cust = st.sidebar.text_input("Khách hàng", "EXPRESS")
files = st.sidebar.file_uploader("Upload", accept_multiple_files=True)

if files and st.sidebar.button("UPLOAD"):
    for f in files:
        d = extract_pdf(f)
        if d["all_specs"]:
            supabase.table("ai_data").insert({
                "file_name":f.name,
                "vector":get_vec(d["img"]),
                "spec_json":d["all_specs"],
                "customer_name":cust
            }).execute()
    st.sidebar.success("Done")

# ================= MAIN =================
st.title("🔍 AI UNIVERSAL AUDITOR")

file = st.file_uploader("Upload audit PDF", type="pdf")

if file:
    d = extract_pdf(file)

    if d["all_specs"]:
        st.success("Đã đọc đúng bảng measurement")

        df = pd.DataFrame(supabase.table("ai_data").select("*").execute().data)

        vec = np.array(get_vec(d["img"])).reshape(1,-1)
        db_vec = np.array(df["vector"].tolist())

        df["sim"] = cosine_similarity(vec, db_vec)[0]
        top = df.sort_values("sim", ascending=False).head(3)

        for _, row in top.iterrows():
            st.subheader(row["file_name"])

            lib = row["spec_json"]
            audit = d["all_specs"]

            s = list(audit.keys())[0]
            lib_s = s if s in lib else list(lib.keys())[0]

            aud_d = {k:v for k,v in audit[s].items() if is_valid_pom(k)}
            lib_d = {k:v for k,v in lib[lib_s].items() if is_valid_pom(k)}

            mapping = match_pom(list(aud_d.keys()), list(lib_d.keys()))

            res = []
            for k,v in aud_d.items():
                mk,score = mapping[k]

                if mk:
                    v2 = lib_d.get(mk,0)
                    diff = round(v - v2, 3)
                    status = "✅" if abs(diff)<0.1 else "❌"
                else:
                    v2 = "-"
                    diff = "-"
                    status = "❌"

                res.append([k,mk,round(score,2),v,v2,diff,status])

            st.table(pd.DataFrame(res,columns=[
                "Audit","Match","Score","Audit Val","Lib Val","Diff","KQ"
            ]))
