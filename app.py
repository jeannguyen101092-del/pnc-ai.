import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, requests
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from streamlit_image_comparison import image_comparison

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Auditor V50", page_icon="🔥")

# ================= MODEL =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

model_ai = load_ai()

# ================= PARSE =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.')
        match = re.findall(r'\d+\.?\d*', txt)
        return float(match[0]) if match else 0
    except:
        return 0

# ================= EXTRACT =================
def extract_pom(pdf_file):
    specs, img_bytes = {}, None
    pdf_bytes = pdf_file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc):
        img_bytes = doc.load_page(0).get_pixmap().tobytes("png")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tb in tables:
                df = pd.DataFrame(tb)
                if len(df.columns) < 2: continue
                for i in range(len(df)):
                    k = str(df.iloc[i][0]).upper()
                    v = parse_val(df.iloc[i][1])
                    if v > 0:
                        specs[k] = v

    return {"specs": specs, "img": img_bytes}

# ================= VECTOR =================
def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        with torch.no_grad():
            v = model_ai(tf(img).unsqueeze(0))
            v = v.view(-1).numpy()
            v = v / np.linalg.norm(v)
        return v.tolist()
    except:
        return None

# ================= SPEC =================
def spec_similarity(s1, s2):
    keys = set(s1) & set(s2)
    if not keys: return 0
    diff = np.mean([abs(s1[k] - s2[k]) for k in keys])
    return max(0, 100 - diff)

# ================= EXPORT =================
def export_excel(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return out.getvalue()

# ================= IMAGE LOAD =================
def load_url_img(url):
    try:
        return Image.open(requests.get(url, stream=True).raw).convert("RGBA")
    except:
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📂 DATA")

    res = supabase.table("ai_data").select("*", count="exact").execute()
    st.metric("Records", res.count)

    files = st.file_uploader("Upload Techpack", accept_multiple_files=True)

    if files and st.button("🚀 NẠP"):
        for f in files:
            d = extract_pom(f)
            vec = get_vector(d['img'])

            img_url = ""
            if d['img']:
                path = f"lib/{f.name}.png"
                try:
                    supabase.storage.from_(BUCKET).upload(path, d['img'], {"upsert":"true"})
                    img_url = supabase.storage.from_(BUCKET).get_public_url(path)["publicUrl"]
                except:
                    pass

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": d['specs'],
                "image_url": img_url
            }).execute()

            st.success(f"✅ {f.name}")

        st.rerun()

# ================= MAIN =================
st.title("🔥 AI Auditor V50 – VISUAL QC TOOL")

file_test = st.file_uploader("Upload file kiểm", type="pdf")

if file_test:
    target = extract_pom(file_test)
    vec_test = get_vector(target['img'])

    db = supabase.table("ai_data").select("*").execute()

    results = []

    for row in db.data:
        sim_img, sim_spec = 0, 0

        if vec_test and row.get("vector"):
            v1 = np.array(vec_test).reshape(1,-1)
            v2 = np.array(row["vector"]).reshape(1,-1)
            sim_img = cosine_similarity(v1, v2)[0][0] * 100

        sim_spec = spec_similarity(target['specs'], row['spec_json'])

        sim = 0.4*sim_img + 0.6*sim_spec

        results.append({"row": row, "sim": sim, "img": sim_img, "spec": sim_spec})

    top = sorted(results, key=lambda x: x['sim'], reverse=True)[:1]

    for r in top:
        st.subheader(f"{r['row']['file_name']} → {r['sim']:.1f}%")
        st.write(f"AI: {r['img']:.1f}% | SPEC: {r['spec']:.1f}%")

        # ===== LOAD IMAGE =====
        img1 = Image.open(io.BytesIO(target['img'])).convert("RGBA") if target['img'] else None
        img2 = load_url_img(r['row'].get("image_url"))

        # ===== MODE =====
        mode = st.radio("Chế độ so sánh", ["Slider", "Overlay", "Diff"])

        if img1 and img2:
            img1 = img1.resize(img2.size)

            if mode == "Slider":
                image_comparison(img1=img1, img2=img2, width=700)

            elif mode == "Overlay":
                alpha = st.slider("Blend", 0.0, 1.0, 0.5)
                blended = Image.blend(img1, img2, alpha)
                st.image(blended, use_column_width=True)

            elif mode == "Diff":
                diff = np.abs(np.array(img1) - np.array(img2))
                st.image(diff, use_column_width=True)

        # ===== SPEC TABLE =====
        diff_data = []
        for k, v in target['specs'].items():
            ref = r['row']['spec_json'].get(k, 0)
            diff_data.append([k, v, ref, round(v-ref,2)])

        df = pd.DataFrame(diff_data, columns=["POM","NEW","REF","DIFF"])
        st.dataframe(df)

        # ===== EXPORT =====
        st.download_button(
            "📥 Export Excel",
            export_excel(df),
            file_name="compare.xlsx"
        )

# ================= RESET =================
if st.button("RESET"):
    st.rerun()
