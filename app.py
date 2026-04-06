import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, gc, json
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET_NAME = "fashion-imgs"

supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V12", page_icon="👔")

# ================= LOAD MODEL (FIXED) =================
@st.cache_resource
def load_ai():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    return model.eval()

ai_brain = load_ai().to("cpu")

# ================= IMAGE FROM EXCEL =================
def excel_to_img_bytes(file_obj):
    try:
        df = pd.read_excel(file_obj).fillna("")
        df_display = df.head(30)
        fig, ax = plt.subplots(figsize=(10, len(df_display) * 0.5))
        ax.axis('off')
        table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        return buf.getvalue()
    except:
        return None

# ================= PARSE =================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\.\d+|\d+)', str(t))
        return float(found[0]) if found else 0
    except:
        return 0

# ================= PDF =================
def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    for r in tb:
                        if not r or len(r) < 2: continue
                        label = str(r[0]).upper()
                        val = parse_val(r[1])
                        if val > 0:
                            specs[label] = val
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "cat": "AUTO"}
    except:
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        db_res = supabase.table("ai_data").select("*").execute()
        all_samples = db_res.data
        st.metric("Tổng mẫu", len(all_samples))
    except:
        all_samples = []
        st.metric("Tổng mẫu", 0)

    files = st.file_uploader("Upload PDF + Excel", accept_multiple_files=True, type=['pdf','xlsx','xls'])

    if files and st.button("🚀 NẠP DATA"):
        for f in files:
            if f.name.endswith(".pdf"):
                with open("tmp.pdf","wb") as t:
                    t.write(f.getbuffer())
                d = get_data("tmp.pdf")
                if d:
                    img = Image.open(io.BytesIO(d['img'])).convert("RGB")
                    tf = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    with torch.no_grad():
                        vec = ai_brain(tf(img).unsqueeze(0)).squeeze().numpy().tolist()

                    buf = io.BytesIO()
                    img.save(buf, format="WEBP")

                    supabase.storage.from_(BUCKET_NAME).upload(f.name+".webp", buf.getvalue(), {"upsert":"true"})
                    url = supabase.storage.from_(BUCKET_NAME).get_public_url(f.name+".webp")

                    supabase.table("ai_data").upsert({
                        "file_name": f.name,
                        "vector": vec,
                        "spec_json": d['spec'],
                        "img_url": url,
                        "category": "ALL"
                    }).execute()

        st.success("DONE")
        st.rerun()

# ================= MAIN =================
st.title("👔 AI Fashion Pro V12")

test_file = st.file_uploader("Upload PDF Test", type="pdf")

if test_file:
    with open("test.pdf","wb") as f:
        f.write(test_file.getbuffer())

    target = get_data("test.pdf")

    if target:
        st.write("DEBUG DB:", len(all_samples))

        img = Image.open(io.BytesIO(target['img'])).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        with torch.no_grad():
            v_test = ai_brain(tf(img).unsqueeze(0)).squeeze().numpy()

        matches = []
        for item in all_samples:
            try:
                v_db = item['vector']
                if isinstance(v_db, str):
                    v_db = json.loads(v_db)
                v_db = np.array(v_db).reshape(1,-1)

                sim = float(cosine_similarity(v_test.reshape(1,-1), v_db))*100
                matches.append(item | {"sim": sim})
            except:
                continue

        if not matches:
            st.warning("❌ Không có dữ liệu match")
        else:
            for m in sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]:
                with st.expander(f"📌 {m['file_name']} - {m['sim']:.1f}%", True):
                    c1,c2,c3 = st.columns([1,1,2])
                    c1.image(target['img'], caption="Test")
                    c2.image(m['img_url'], caption="DB")

                    # ===== SO SÁNH THÔNG SỐ =====
                    res = []
                    for kt, vt in target['spec'].items():
                        mk = next((k for k in m['spec_json'].keys() if SequenceMatcher(None, kt, k).ratio() > 0.8), None)
                        vd = m['spec_json'][mk] if mk else 0
                        res.append({
                            "Thông số": kt,
                            "Test": vt,
                            "Kho": vd,
                            "Lệch": round(vt - vd,2)
                        })

                    df = pd.DataFrame(res)
                    c3.dataframe(df, use_container_width=True)

                    # ===== XUẤT EXCEL =====
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)

                    c3.download_button(
                        "📥 Xuất Excel so sánh",
                        out.getvalue(),
                        file_name=f"SoSanh_{m['file_name']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if os.path.exists("test.pdf"):
    os.remove("test.pdf")

gc.collect()
