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

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V48", page_icon="🔥")

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
        txt = str(t).replace(',', '.')
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

    # COUNT
    try:
        count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("📦 Số mẫu", count.count)
    except:
        st.warning("DB lỗi")

    # CLEAN
    if st.button("🧹 Clean DB lỗi"):
        db = supabase.table("ai_data").select("*").execute()
        for item in db.data:
            if not item.get("vector") or len(item["vector"])!=512:
                supabase.table("ai_data").delete().eq("id", item["id"]).execute()
        st.success("Done!")

    # UPLOAD
    files = st.file_uploader("Upload mẫu", accept_multiple_files=True)

    if files and st.button("🚀 Nạp kho"):
        for f in files:
            data = extract_pdf(f)
            vec = get_vector(data["img"])

            if not vec:
                st.error(f"Vector lỗi: {f.name}")
                continue

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
st.title("🔍 AI Fashion Auditor V48")

# ===== DATA LIBRARY =====
st.subheader("📦 DATA LIBRARY")

db = supabase.table("ai_data").select("*").execute()
data_list = db.data if db.data else []

col1, col2 = st.columns([1,2])

with col1:
    st.metric("Tổng mẫu", len(data_list))

with col2:
    keyword = st.text_input("🔍 Tìm mẫu")

if keyword:
    data_list = [d for d in data_list if keyword.lower() in d["file_name"].lower()]

cols = st.columns(4)

for i, item in enumerate(data_list):
    with cols[i % 4]:
        st.image(item["image_url"], use_container_width=True)
        st.caption(item["file_name"])

        if st.button("Xem", key=f"view_{i}"):
            st.session_state["selected"] = item

# ===== DETAIL =====
if "selected" in st.session_state:
    sel = st.session_state["selected"]

    st.subheader("📄 Chi tiết mẫu")
    c1, c2 = st.columns(2)

    c1.image(sel["image_url"])
    c2.write("📌 File:", sel["file_name"])
    c2.write("📊 POM:", len(sel["spec_json"]))

    if st.button("Đóng"):
        del st.session_state["selected"]

# ===== AUDIT =====
st.subheader("🔎 KIỂM TRA")

file = st.file_uploader("Upload file kiểm tra", type="pdf")

if file:
    data = extract_pdf(file)
    vec_test = get_vector(data["img"])

    if not vec_test:
        st.error("Vector lỗi")
        st.stop()

    vec_test = np.array(vec_test).reshape(1,-1)

    matches=[]
    for item in db.data:
        vec_db = item.get("vector")
        if not vec_db or len(vec_db)<100:
            continue

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

    total=len(df)
    fail=len(df[df["Result"]=="FAIL"])

    c1,c2,c3 = st.columns(3)
    c1.metric("Tổng POM", total)
    c2.metric("Lỗi", fail)
    c3.metric("Tỷ lệ lỗi", f"{fail/total*100:.1f}%")

    st.dataframe(df, use_container_width=True)

    out = io.BytesIO()
    df.to_excel(out, index=False)
    st.download_button("📥 Export Excel", out.getvalue(), "audit.xlsx")
