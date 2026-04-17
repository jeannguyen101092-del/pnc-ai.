```python
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. AI CORE (NÂNG CẤP) =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()
model_ai = load_model()

def get_vector(img_bytes):
    if not img_bytes: return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.0).convert('RGB')

        w, h = img.size

        top = img.crop((0, 0, w, int(h*0.3)))
        mid = img.crop((0, int(h*0.3), w, int(h*0.65)))
        bot = img.crop((0, int(h*0.65), w, h))

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        def encode(im):
            with torch.no_grad():
                v = model_ai(tf(im).unsqueeze(0)).flatten().cpu().numpy()
                n = np.linalg.norm(v)
                return (v/n).astype(float) if n > 0 else v

        return {
            "top": encode(top),
            "mid": encode(mid),
            "bot": encode(bot)
        }
    except:
        return None

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "page", "tol", "+", "-"]): return 0
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        return float(num[0]) if num else 0
    except: return 0

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 3. SCRAPER =================
def extract_full_data(file_content):
    if not file_content: return None
    all_specs, img_bytes = {}, None
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        buf = io.BytesIO(); img_pil.save(buf, format="WEBP", quality=70); img_bytes = buf.getvalue(); doc.close()

        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words: continue
                for w in words:
                    txt = w['text']
                    if len(txt) > 3:
                        if "ALL" not in all_specs:
                            all_specs["ALL"] = {}
                        all_specs["ALL"][txt] = parse_val(txt)
        return {"all_specs": all_specs, "img": img_bytes}
    except: return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        for f in new_files:
            data = extract_full_data(f.getvalue())
            if data and data['img']:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})

                vecs = get_vector(data['img'])

                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)),
                    "file_name": f.name,
                    "vector": vecs["mid"].tolist() if vecs else None,
                    "vec_top": vecs["top"].tolist() if vecs else None,
                    "vec_mid": vecs["mid"].tolist() if vecs else None,
                    "vec_bot": vecs["bot"].tolist() if vecs else None,
                    "spec_json": data['all_specs'],
                    "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
        st.success("Done!")
        st.rerun()

# ================= 5. MAIN =================
st.title("AI AUDITOR")

f_audit = st.file_uploader("Upload Target PDF:", type="pdf")

if f_audit:
    target = extract_full_data(f_audit.getvalue())
    if target and target['img']:
        target_name = f_audit.name.upper()

        res = supabase.table("ai_data").select("id, vec_top, vec_mid, vec_bot, file_name").execute()

        t_vecs = get_vector(target['img'])
        valid_rows = []

        for r in res.data:
            try:
                if not r.get('vec_mid'): continue

                mid_db = np.array(r['vec_mid']).reshape(1, -1)
                top_db = np.array(r['vec_top']).reshape(1, -1) if r.get('vec_top') else mid_db
                bot_db = np.array(r['vec_bot']).reshape(1, -1) if r.get('vec_bot') else mid_db

                mid_t = np.array(t_vecs['mid']).reshape(1, -1)
                top_t = np.array(t_vecs['top']).reshape(1, -1)
                bot_t = np.array(t_vecs['bot']).reshape(1, -1)

                sim = (
                    0.5 * cosine_similarity(mid_t, mid_db)[0][0] +
                    0.3 * cosine_similarity(bot_t, bot_db)[0][0] +
                    0.2 * cosine_similarity(top_t, top_db)[0][0]
                )

                if "PANT" in target_name and "PANT" in r['file_name'].upper():
                    sim += 0.2

                r['sim_final'] = sim
                valid_rows.append(r)
            except:
                continue

        df = pd.DataFrame(valid_rows).sort_values("sim_final", ascending=False).head(3)

        for _, row in df.iterrows():
            st.write(row['file_name'], f"{row['sim_final']:.2%}")
```
