import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI AUDITOR V97")

if "up_key" not in st.session_state:
    st.session_state.up_key = 0

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= TOOLS =================
def parse_val(t):
    try:
        t = str(t).replace(",", ".")
        m = re.findall(r"\d+\.\d+|\d+", t)
        if not m: return 0
        v = float(m[0])
        if v > 200: return 0
        return v
    except:
        return 0

def detect_category(text):
    t = text.upper()
    if any(x in t for x in ["DRESS","SKIRT","VÁY"]): return "VÁY"
    if any(x in t for x in ["PANT","JEAN","QUẦN"]): return "QUẦN"
    if any(x in t for x in ["SHIRT","JACKET","ÁO"]): return "ÁO"
    return "KHÁC"

def extract_customer(text):
    t = text.upper()
    m = re.search(r'(CUSTOMER|CLIENT|BRAND)[:\s]+([A-Z0-9 _-]+)', t)
    return m.group(2).strip() if m else "UNKNOWN"

def extract_size(text):
    m = re.search(r'\b(XXS|XS|S|M|L|XL|XXL|\d{1,2})\b', text.upper())
    return m.group(1) if m else "UNKNOWN"

def get_vec(img):
    img = Image.open(io.BytesIO(img)).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        v = model_ai(tf(img).unsqueeze(0)).flatten().numpy()
    return v.tolist()

# ================= EXTRACT =================
def extract_pdf(file):
    try:
        file.seek(0)
        content = file.read()

        doc = fitz.open(stream=content, filetype="pdf")
        img = doc[0].get_pixmap().tobytes("png")
        text = " ".join([p.get_text() for p in doc])
        doc.close()

        category = detect_category(text)
        customer = extract_customer(text)
        size = extract_size(text)

        specs = {}

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    if len(df.columns) < 2: continue

                    name_col = 0
                    val_col = 1

                    for i in range(len(df.columns)):
                        nums = sum(1 for v in df.iloc[:10,i] if parse_val(v)>0)
                        if nums > 3:
                            val_col = i

                    for i in range(len(df)):
                        name = str(df.iloc[i,name_col]).upper().strip()
                        val = parse_val(df.iloc[i,val_col])
                        if len(name)>3 and val>0:
                            specs[name]=val

                if specs: break

        return {
            "specs": specs,
            "img": img,
            "category": category,
            "customer": customer,
            "size": size
        }
    except:
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.title("KHO")

    try:
        count = supabase.table("ai_data").select("*", count="exact").execute().count
        st.metric("Tổng mẫu", count or 0)
    except:
        st.error("DB lỗi")

    files = st.file_uploader("Upload", accept_multiple_files=True, key=st.session_state.up_key)

    if files and st.button("NẠP"):
        for f in files:
            data = extract_pdf(f)
            if not data: continue

            vec = get_vec(data["img"])
            name = re.sub(r'[^a-zA-Z0-9]', '_', f.name)

            path = f"{name}.png"

            supabase.storage.from_(BUCKET).upload(path, data["img"], {"upsert":"true"})
            url = supabase.storage.from_(BUCKET).get_public_url(path)

            supabase.table("ai_data").insert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": data["specs"],
                "image_url": url,
                "category": data["category"],
                "customer": data["customer"],
                "size": data["size"]
            }).execute()

        st.success("OK")
        st.session_state.up_key += 1
        st.rerun()

# ================= AUDIT =================
st.title("AI AUDITOR")

file = st.file_uploader("Upload audit", type="pdf")
customer_filter = st.text_input("Customer filter")

if file:
    target = extract_pdf(file)

    if not target:
        st.error("Lỗi đọc file")
    else:
        st.write(target["category"], target["customer"], target["size"])

        q = supabase.table("ai_data").select("*").eq("category", target["category"])

        if customer_filter:
            q = q.ilike("customer", f"%{customer_filter.upper()}%")

        res = q.execute().data

        if not res:
            st.warning("Không có data")
        else:
            tvec = np.array(get_vec(target["img"])).reshape(1,-1)

            scores = []
            for r in res:
                vec = np.array(r["vector"]).reshape(1,-1)
                sim = cosine_similarity(tvec, vec)[0][0]

                size_bonus = 1 if r.get("size")==target["size"] else 0.6

                scores.append({**r, "score": sim*size_bonus})

            top = sorted(scores, key=lambda x:x["score"], reverse=True)[:3]

            cols = st.columns(3)
            for i,t in enumerate(top):
                with cols[i]:
                    st.image(t["image_url"], caption=f"{t['file_name']} {t['score']:.2f}")

            sel = st.selectbox("Chọn mẫu", [t["file_name"] for t in top])
            best = next(x for x in top if x["file_name"]==sel)

            rows=[]
            for k,v in target["specs"].items():
                m = best["spec_json"].get(k,0)
                diff = v-m if m else 0

                if m==0: status="MISS"
                elif abs(diff)>0.125: status="LECH"
                else: status="OK"

                rows.append([k,v,m,diff,status])

            df = pd.DataFrame(rows, columns=["POM","NEW","OLD","DIFF","KQ"])
            st.dataframe(df, use_container_width=True)

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                df.to_excel(w, index=False)

            st.download_button("DOWNLOAD", buf.getvalue(), "audit.xlsx")
