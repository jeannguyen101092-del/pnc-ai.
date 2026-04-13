import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np, requests
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V47", page_icon="🔥")

# ================= INIT =================
supabase = create_client(URL, KEY)

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= VECTOR =================
@st.cache_data
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0))
            vec = vec.view(-1).cpu().numpy()

        return vec.tolist() if len(vec)==512 else None
    except:
        return None

# ================= UTILS =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').lower()
        match = re.findall(r'(\d+\.\d+|\d+)', txt)
        return float(match[0]) if match else 0
    except:
        return 0

def clean_key(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT =================
def extract_pdf(file):
    specs, img = {}, None
    file.seek(0)
    pdf_bytes = file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc)>0:
        img = doc[0].get_pixmap().tobytes("png")
    doc.close()

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for tb in page.extract_tables():
                df = pd.DataFrame(tb)
                if df.empty: continue

                for _, row in df.iterrows():
                    name = str(row[0]).upper()
                    if len(name)<3: continue

                    for c in row[1:]:
                        val = parse_val(c)
                        if val>0:
                            specs[name]=val
                            break

    return {"specs": specs, "img": img}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ SETTINGS")

    tol = st.selectbox("Tolerance", [0.1,0.2,0.5,1.0], index=1)

    if st.button("🧹 Clean DB lỗi"):
        db = supabase.table("ai_data").select("*").execute()
        for item in db.data:
            if not item.get("vector") or len(item["vector"])!=512:
                supabase.table("ai_data").delete().eq("id", item["id"]).execute()
        st.success("Done!")

    files = st.file_uploader("Upload mẫu", accept_multiple_files=True)

    if files and st.button("🚀 Nạp kho"):
        for f in files:
            data = extract_pdf(f)
            vec = get_vector(data["img"])

            if not vec: continue

            path = f"{f.name}.png"
            supabase.storage.from_(BUCKET).upload(path, data["img"], {"upsert":"true"})

            url = supabase.storage.from_(BUCKET).get_public_url(path)

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": data["specs"],
                "image_url": url
            }).execute()

            st.success(f"✔ {f.name}")

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V47")

file = st.file_uploader("Upload file kiểm tra", type="pdf")

if file:
    data = extract_pdf(file)
    vec_test = get_vector(data["img"])
    vec_test = np.array(vec_test).reshape(1,-1)

    db = supabase.table("ai_data").select("*").execute()

    matches=[]
    for item in db.data:
        vec_db = item.get("vector")
        if not vec_db or len(vec_db)!=512: continue

        v_ref = np.array(vec_db).reshape(1,-1)
        score = cosine_similarity(vec_test, v_ref)[0][0]

        matches.append({"item":item,"score":score})

    if not matches:
        st.error("Không có dữ liệu")
        st.stop()

    matches = sorted(matches, key=lambda x:x["score"], reverse=True)[:5]

    st.subheader("🏆 TOP MATCH")

    for m in matches:
        st.write(f"{m['item']['file_name']} → {m['score']*100:.2f}%")

    best = matches[0]["item"]

    st.subheader("📊 SO SÁNH CHI TIẾT")

    ref_map = {clean_key(k):v for k,v in best["spec_json"].items()}

    rows=[]
    for k,v in data["specs"].items():
        ref = ref_map.get(clean_key(k),0)
        diff = v-ref

        rows.append({
            "POM":k,
            "Target":v,
            "Ref":ref,
            "Diff":round(diff,3),
            "Result":"OK" if abs(diff)<=tol else "FAIL"
        })

    df = pd.DataFrame(rows)

    # ===== KPI =====
    total = len(df)
    fail = len(df[df["Result"]=="FAIL"])

    c1,c2,c3 = st.columns(3)
    c1.metric("Tổng POM", total)
    c2.metric("Lỗi", fail)
    c3.metric("Tỷ lệ lỗi", f"{fail/total*100:.1f}%")

    st.dataframe(df, use_container_width=True)

    # EXPORT
    out = io.BytesIO()
    df.to_excel(out, index=False)
    st.download_button("📥 Export Excel", out.getvalue(), "audit.xlsx")
