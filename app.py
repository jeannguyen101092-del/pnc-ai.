import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
GH_TOKEN = "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"
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V9")

# ================= AI =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= UPLOAD GITHUB =================
def upload_to_github(img_bytes, filename):
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        url = f"https://api.github.com/repos/{GH_REPO}/contents/imgs/{clean_name}.jpg"

        headers = {
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        content = base64.b64encode(img_bytes).decode("utf-8")

        # check tồn tại
        r = requests.get(url, headers=headers)
        data = {"message": "upload", "content": content, "branch": GH_BRANCH}

        if r.status_code == 200:
            data["sha"] = r.json()["sha"]

        res = requests.put(url, headers=headers, json=data)

        if res.status_code in [200, 201]:
            return f"https://raw.githubusercontent.com/{GH_REPO}/{GH_BRANCH}/imgs/{clean_name}.jpg"
        else:
            st.error(res.text)
            return None
    except Exception as e:
        st.error(e)
        return None

# ================= PARSE =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# ================= CLASSIFY =================
def classify(text):
    t = text.upper()
    if "CARGO" in t: return "QUAN CARGO"
    if "SHORT" in t: return "QUAN SHORT"
    if "PANT" in t: return "QUAN DAI"
    if "SHIRT" in t: return "AO SO MI"
    if "DRESS" in t: return "DAM"
    return "AO"

# ================= EXTRACT =================
def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    for r in tb:
                        if not r: continue
                        vals = [parse_val(x) for x in r if x]
                        if vals: specs[str(r[0])] = np.mean(vals)

        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap().tobytes("png")
        doc.close()

        return {"spec": specs, "img": img, "cat": classify(text)}
    except:
        return None

# ================= UI =================
st.title("AI Fashion Pro V9")

files = st.file_uploader("Upload PDF", accept_multiple_files=True)

if files and st.button("NAP KHO"):
    for f in files:
        with open("tmp.pdf", "wb") as t:
            t.write(f.getbuffer())

        d = get_data("tmp.pdf")
        if not d or len(d['spec']) == 0:
            st.warning(f"Bo qua: {f.name}")
            continue

        img_pil = Image.open(io.BytesIO(d['img'])).convert("RGB")
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=80)

        url = upload_to_github(buf.getvalue(), f.name)

        if url:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            with torch.no_grad():
                vec = ai_brain(tf(img_pil).unsqueeze(0)).flatten().numpy().tolist()

            supabase.table("ai_data").upsert({
                "file_name": f.name,
                "vector": vec,
                "spec_json": d['spec'],
                "img_url": url,
                "category": d['cat']
            }, on_conflict="file_name").execute()

            st.success(f"OK: {f.name}")

        os.remove("tmp.pdf")

# ================= TEST =================
test = st.file_uploader("TEST PDF")

if test:
    with open("test.pdf", "wb") as f:
        f.write(test.getbuffer())

    t = get_data("test.pdf")

    st.image(t['img'])
    st.write(t['cat'])

    db = supabase.table("ai_data").select("*").eq("category", t['cat']).execute()

    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    with torch.no_grad():
        v_test = ai_brain(tf(Image.open(io.BytesIO(t['img']))).unsqueeze(0)).flatten().numpy()

    results = []

    for item in db.data:
        v_db = np.array(item['vector'])
        sim = cosine_similarity(v_test.reshape(1,-1), v_db.reshape(1,-1))[0][0] * 100
        results.append((item['file_name'], sim, item['img_url']))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    for r in results:
        st.image(r[2])
        st.write(r[0], f"{r[1]:.1f}%")
