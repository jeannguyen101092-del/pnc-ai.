import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V44", page_icon="📊")

if "up_key" not in st.session_state: st.session_state.up_key = 0
if "au_key" not in st.session_state: st.session_state.au_key = 0

# --- LOAD AI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_ai()

# --- PARSE VALUE ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if not txt or txt in ['nan', '-', 'none']: return 0
        match = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not match: return 0
        v = match[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except:
        try:
            return float(str(t).replace('"','').strip())
        except:
            return 0

def clean_pos(t):
    return re.sub(r'[^A-Z0-9]', '', str(t).upper())

# --- CORE AI EXTRACT V44 ---
def extract_pom_v44(pdf_file):
    full_specs, img_bytes, brand = {}, None, "OTHER"

    try:
        pdf_content = pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")

        # --- IMG ---
        if len(doc) > 0:
            img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")

        # --- TEXT ---
        all_text = ""
        for p in doc:
            all_text += (p.get_text() or "").upper() + " "

        if "REITMANS" in all_text:
            brand = "REITMANS"
        doc.close()

        # --- KEYWORDS ---
        POM_KEYS = ["POM", "POINT OF MEASURE", "MEASUREMENT", "DIMENSION"]
        VALUE_KEYS = ["NEW", "SPEC", "MEAS", "GARMENT", "SAMPLE"]

        SIZE_KEYS = ["XXS","XS","S","M","L","XL","XXL","XXXL","2XL","3XL","4XL"]

        BOM_KEYS = ["FABRIC","TRIM","THREAD","POCKET","ZIPPER","BUTTON","POLY","COTTON"]

        # --- READ PDF ---
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:

                text_page = (page.extract_text() or "").upper()

                # ❌ skip BOM
                if any(k in text_page for k in BOM_KEYS):
                    continue

                # ✅ only POM page
                if not any(k in text_page for k in POM_KEYS):
                    continue

                tables = page.extract_tables()

                for tb in tables:
                    df = pd.DataFrame(tb)
                    if df.empty or len(df.columns) < 2:
                        continue

                    df = df.fillna("").astype(str)

                    header_row = -1
                    pom_idx = -1
                    val_idx = -1

                    # --- FIND HEADER ---
                    for r_idx, row in df.head(10).iterrows():
                        row_up = [str(c).upper().strip() for c in row]

                        for i, c in enumerate(row_up):
                            if any(k in c for k in POM_KEYS):
                                pom_idx = i

                        # --- AUTO DETECT VALUE COLUMN ---
                        # 1. NEW/SPEC
                        for i, c in enumerate(row_up):
                            if any(k in c for k in VALUE_KEYS):
                                val_idx = i
                                break

                        # 2. SIZE M
                        if val_idx == -1:
                            for i, c in enumerate(row_up):
                                if c == "M":
                                    val_idx = i
                                    break

                        # 3. ANY SIZE
                        if val_idx == -1:
                            for i, c in enumerate(row_up):
                                if c in SIZE_KEYS:
                                    val_idx = i
                                    break

                        if pom_idx != -1 and val_idx != -1:
                            header_row = r_idx
                            break

                    # fallback
                    if header_row == -1:
                        pom_idx = 0
                        val_idx = 1

                    # --- READ DATA ---
                    for i in range(header_row + 1, len(df)):
                        row = df.iloc[i]

                        try:
                            name = str(row[pom_idx]).replace("\n"," ").strip().upper()
                            val_raw = row[val_idx]

                            if len(name) < 3:
                                continue

                            # ❌ skip BOM line
                            if any(k in name for k in BOM_KEYS):
                                continue

                            # ❌ skip noise
                            if any(x in name for x in ["REF","NOTE","GRADE","TOL"]):
                                continue

                            val = parse_val(val_raw)

                            if val > 0:
                                full_specs[name] = val

                        except:
                            continue

        return {"specs": full_specs, "img": img_bytes, "brand": brand}

    except Exception as e:
        print("ERROR:", e)
        return None


# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 KHO DỮ LIỆU CHUẨN")
    res_c = supabase.table("ai_data").select("file_name", count="exact").execute()
    st.metric("Mẫu đã nạp", res_c.count if res_c.count else 0)

    files = st.file_uploader("Nạp Techpack mẫu", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")

    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p = st.progress(0)
        for i, f in enumerate(files):

            d = extract_pom_v44(f)
            if d and d['specs']:

                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')

                tf = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])

                with torch.no_grad():
                    vec = model_ai(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()

                path = f"lib_{f.name.replace('.pdf','.png')}"

                supabase.storage.from_(BUCKET).upload(
                    path=path,
                    file=d['img'],
                    file_options={"content-type":"image/png","x-upsert":"true"}
                )

                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": d['specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path),
                    "category": d['brand']
                }).execute()

            p.progress((i+1)/len(files))

        st.session_state.up_key += 1
        st.rerun()

# --- MAIN ---
st.title("🔍 AI Fashion Auditor V44")

t_file = st.file_uploader("Upload file đối soát", type="pdf", key=f"a_{st.session_state.au_key}")

if t_file:
    target = extract_pom_v44(t_file)

    if target and target['specs']:
        st.success(f"✅ Tìm thấy {len(target['specs'])} thông số")

        db_res = supabase.table("ai_data").select("*").execute()

        if db_res.data:

            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')

            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            with torch.no_grad():
                v_test = model_ai(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []

            for i in db_res.data:
                if i.get('vector'):
                    v_ref = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                    matches.append({"data": i, "sim": sim})

            top = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]

            for m in top:
                st.subheader(f"✨ Match: {m['data']['file_name']} ({m['sim']:.1f}%)")

                c1, c2 = st.columns(2)
                with c1: st.image(target['img'])
                with c2: st.image(m['data']['image_url'])

                diff_list = []

                for p_name, v_target in target['specs'].items():
                    v_ref = 0
                    p_clean = clean_pos(p_name)

                    for k_ref, val_ref in m['data']['spec_json'].items():
                        if p_clean == clean_pos(k_ref):
                            v_ref = val_ref
                            break

                    if v_target > 0 or v_ref > 0:
                        diff_list.append({
                            "POM": p_name,
                            "Target": v_target,
                            "Ref": v_ref,
                            "Diff": round(v_target - v_ref, 2)
                        })

                if diff_list:
                    df = pd.DataFrame(diff_list)
                    st.dataframe(df)

                    out = io.BytesIO()
                    df.to_excel(out, index=False)

                    st.download_button("📥 Excel", out.getvalue(), "report.xlsx")

    else:
        st.error("Không đọc được POM")
