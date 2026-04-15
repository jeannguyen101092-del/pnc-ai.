import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI Universal Auditor V114", page_icon="🏢")

if 'up_key' not in st.session_state:
    st.session_state.up_key = 0

# ================= LOAD MODEL =================
@st.cache_resource
def load_models():
    cnn = models.resnet18(pretrained=True)
    cnn.fc = torch.nn.Identity()
    cnn.eval()

    nlp = SentenceTransformer('all-MiniLM-L6-v2')
    return cnn, nlp

model_ai, nlp_model = load_models()

# ================= CATEGORY =================
def smart_detect_category(text):
    t = text.upper()
    if any(x in t for x in ["INSEAM","THIGH"]): return "QUẦN"
    if any(x in t for x in ["CHEST","SLEEVE"]): return "ÁO"
    if "DRESS" in t: return "VÁY"
    return "KHÁC"

# ================= POM ENGINE =================
def normalize_pom(p):
    p = str(p).upper()

    rules = {
        "WAIST":["WAIST","WAISTBAND"],
        "HIP":["HIP","SEAT"],
        "THIGH":["THIGH"],
        "INSEAM":["INSEAM"],
        "OUTSEAM":["OUTSEAM"],
        "LEG OPENING":["LEG OPEN"],
        "CHEST":["CHEST","BUST"],
        "SLEEVE":["SLEEVE"],
        "LENGTH":["LENGTH"]
    }

    for k,v in rules.items():
        if any(x in p for x in v):
            return k
    return p

def is_valid_pom(p):
    p = str(p).upper()
    blacklist = ["DESCRIPTION","STYLE","COLOR","DATE","SEASON"]
    if any(b in p for b in blacklist):
        return False
    return len(p) > 3

# ================= VALUE =================
def parse_val(t):
    try:
        txt = str(t).replace(",",".")
        m = re.search(r'(\d+\.\d+|\d+)', txt)
        return float(m.group(1)) if m else 0
    except:
        return 0

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

# ================= AI MATCH =================
def match_pom(aud_keys, lib_keys):
    aud = [normalize_pom(k) for k in aud_keys]
    lib = [normalize_pom(k) for k in lib_keys]

    emb_a = nlp_model.encode(aud)
    emb_b = nlp_model.encode(lib)

    sim = cosine_similarity(emb_a, emb_b)

    mapping = {}
    for i,a in enumerate(aud):
        idx = np.argmax(sim[i])
        score = sim[i][idx]
        mapping[aud_keys[i]] = (lib_keys[idx], score) if score>0.65 else (None, score)

    return mapping

# ================= EXTRACT =================
def extract_pdf(file):
    all_specs, img_bytes = {}, None

    file.seek(0)
    pdf_content = file.read()

    doc = fitz.open(stream=pdf_content, filetype="pdf")
    img_bytes = doc.load_page(0).get_pixmap().tobytes("png")
    full_text = " ".join([p.get_text() for p in doc])
    doc.close()

    category = smart_detect_category(full_text)

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
                        if i!=n_col and len(v)<6:
                            size_cols[i]=v

                    if n_col!=-1 and size_cols:
                        break

                if n_col==-1:
                    continue

                for s_col,s_name in size_cols.items():
                    all_specs.setdefault(s_name,{})
                    for r in range(len(df)):
                        raw = df.iloc[r,n_col]
                        pom = normalize_pom(raw)
                        val = parse_val(df.iloc[r,s_col])

                        if is_valid_pom(pom) and val>0:
                            all_specs[s_name][pom]=val

            if all_specs:
                break

    return {"all_specs":all_specs,"img":img_bytes,"category":category}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO")
    cust = st.text_input("Khách hàng", "EXPRESS")
    files = st.file_uploader("Upload", accept_multiple_files=True)

    if files and st.button("UPLOAD"):
        for f in files:
            d = extract_pdf(f)
            if d["all_specs"]:
                supabase.table("ai_data").insert({
                    "file_name":f.name,
                    "vector":get_vec(d["img"]),
                    "spec_json":d["all_specs"],
                    "category":d["category"],
                    "customer_name":cust
                }).execute()
        st.success("Done")

# ================= MAIN =================
st.title("🔍 AI UNIVERSAL AUDITOR")

file = st.file_uploader("Upload audit file", type="pdf")

if file:
    d = extract_pdf(file)

    if d["all_specs"]:
        st.success("Đã đọc file")

        res = supabase.table("ai_data").select("*").execute()
        df = pd.DataFrame(res.data)

        vec = np.array(get_vec(d["img"])).reshape(1,-1)
        db_vec = np.array(df["vector"].tolist())

        df["sim"]=cosine_similarity(vec,db_vec)[0]
        top=df.sort_values("sim",ascending=False).head(3)

        for _,row in top.iterrows():
            st.subheader(row["file_name"])

            lib=row["spec_json"]
            audit=d["all_specs"]

            s=list(audit.keys())[0]
            lib_s = s if s in lib else list(lib.keys())[0]

            aud_d = audit[s]
            lib_d = lib[lib_s]

            mapping = match_pom(list(aud_d.keys()), list(lib_d.keys()))

            res_l=[]
            for k,v in aud_d.items():
                mk,score = mapping[k]

                if mk:
                    v2=lib_d.get(mk,0)
                    diff=round(v-v2,3)
                    status="✅" if abs(diff)<0.1 else "❌"
                else:
                    v2="-"
                    diff="-"
                    status="❌"

                res_l.append([k,mk,round(score,2),v,v2,diff,status])

            st.table(pd.DataFrame(res_l,columns=[
                "Audit POM","Match","Score","Audit","Lib","Diff","KQ"
            ]))
