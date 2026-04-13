import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V45", page_icon="🔥")

# ================= INIT =================
@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

try:
    supabase: Client = init_supabase()
except:
    st.error("❌ Chưa cấu hình Supabase")
    st.stop()

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

# ================= UTILS =================
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if txt in ['', 'nan', '-', 'none']: return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            a, b = v.split()
            return float(a) + eval(b)
        return eval(v) if '/' in v else float(v)
    except:
        return 0

def clean_key(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# ================= EXTRACT PDF =================
def extract_pdf(file):
    specs, img_bytes, brand = {}, None, "OTHER"

    try:
        file.seek(0)
        pdf_bytes = file.read()

        # IMAGE
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) > 0:
            pix = doc[0].get_pixmap()
            img_bytes = pix.tobytes("png")

        text_all = ""
        for p in doc:
            text_all += (p.get_text() or "").upper()

        if "REITMANS" in text_all:
            brand = "REITMANS"

        doc.close()

        # TABLE
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2:
                        continue

                    for i in range(len(df)):
                        row = df.iloc[i]

                        name = str(row[0]).strip().upper()
                        if len(name) < 3:
                            continue

                        val = 0
                        for c in row[1:]:
                            v = parse_val(c)
                            if v > 0:
                                val = v
                                break

                        if val > 0:
                            specs[name] = val

        return {"specs": specs, "img": img_bytes, "brand": brand}

    except Exception as e:
        st.error(f"Lỗi extract: {e}")
        return None

# ================= VECTOR =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        with torch.no_grad():
            vec = model_ai(tf(img).unsqueeze(0)).flatten().numpy()

        return vec.tolist()
    except:
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 KHO DATA")

    try:
        count = supabase.table("ai_data").select("id", count="exact").execute()
        st.metric("Tổng mẫu", count.count)
    except:
        st.warning("Chưa có DB")

    files = st.file_uploader("Upload mẫu", type="pdf", accept_multiple_files=True)

    if files and st.button("🚀 Nạp kho"):
        for f in files:
            try:
                # check trùng
                check = supabase.table("ai_data").select("id").eq("file_name", f.name).execute()
                if check.data:
                    st.warning(f"Trùng: {f.name}")
                    continue

                data = extract_pdf(f)
                if not data or not data["specs"]:
                    st.error(f"Lỗi đọc: {f.name}")
                    continue

                vec = get_vector(data["img"])
                if not vec or len(vec) != 512:
                    st.error(f"Vector lỗi: {f.name}")
                    continue

                path = f"{f.name}.png"

                supabase.storage.from_(BUCKET).upload(
                    path=path,
                    file=data["img"],
                    file_options={"content-type": "image/png", "upsert": "true"}
                )

                img_url = supabase.storage.from_(BUCKET).get_public_url(path)

                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": data["specs"],
                    "image_url": img_url,
                    "category": data["brand"]
                }).execute()

                st.success(f"✔ {f.name}")

            except Exception as e:
                st.error(f"Lỗi {f.name}: {e}")

# ================= MAIN =================
st.title("🔍 AI Fashion Auditor V45")

file = st.file_uploader("Upload file kiểm tra", type="pdf")

if file:
    data = extract_pdf(file)

    if not data or not data["specs"]:
        st.error("❌ Không đọc được file")
        st.stop()

    st.success(f"✔ Tìm thấy {len(data['specs'])} thông số")

    db = supabase.table("ai_data").select("*").execute()

    if not db.data:
        st.warning("Kho rỗng")
        st.stop()

    vec_test = np.array(get_vector(data["img"])).reshape(1, -1)

    matches = []

    for item in db.data:
        try:
            vec_db = item.get("vector", None)

            if not vec_db or len(vec_db) != 512:
                continue

            v_ref = np.array(vec_db).reshape(1, -1)

            if vec_test.shape[1] != v_ref.shape[1]:
                continue

            score = cosine_similarity(vec_test, v_ref)[0][0]

            matches.append({
                "data": item,
                "score": score
            })

        except:
            continue

    if not matches:
        st.error("❌ Không có dữ liệu hợp lệ để so sánh")
        st.stop()

    best = sorted(matches, key=lambda x: x["score"], reverse=True)[0]

    st.subheader(f"🎯 Match: {best['data']['file_name']} ({best['score']*100:.2f}%)")

    col1, col2 = st.columns(2)
    col1.image(data["img"], caption="File kiểm")
    col2.image(best["data"]["image_url"], caption="File gốc")

    # ===== COMPARE =====
    ref_map = {clean_key(k): v for k, v in best["data"]["spec_json"].items()}

    rows = []
    for k, v in data["specs"].items():
        ref = ref_map.get(clean_key(k), 0)
        diff = round(v - ref, 3)

        rows.append({
            "POM": k,
            "Target": v,
            "Ref": ref,
            "Diff": diff,
            "Result": "OK" if abs(diff) < 0.001 else "LECH"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # EXPORT
    out = io.BytesIO()
    df.to_excel(out, index=False)

    st.download_button("📥 Export Excel", out.getvalue(), "audit.xlsx")
