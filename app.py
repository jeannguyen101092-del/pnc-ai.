# 🚀 V37 PRO MAX - AI GARMENT ANALYZER (IMAGE + POM)

import streamlit as st
import io, fitz, pdfplumber
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
import re

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(URL, KEY)
BUCKET = "images"

# ================= MODEL (STRONGER) =================
@st.cache_resource

def load_model():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= FEATURE EXTRACTION =================

def get_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(img_t).numpy().flatten()
    return vec / np.linalg.norm(vec)

# ================= PDF PARSER V37 =================

def extract_pdf_advanced(pdf_file):
    pdf_bytes = pdf_file.read()

    img_design = None
    pom_img = None
    pom_data = {}
    pom_page_index = None

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").upper()

            # detect design page
            if any(k in text for k in ["SKETCH","DESIGN","DETAIL"]):
                if img_design is None:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    p = doc.load_page(i)
                    img_design = p.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
                    doc.close()

            # detect POM
            if "POM DESCRIPTION" in text and "+TOL" in text:
                pom_page_index = i

                table = page.extract_table()
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    pom_data = df.to_dict()

    if pom_page_index is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        p = doc.load_page(pom_page_index)
        pom_img = p.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

    return img_design, pom_img, pom_data

# ================= UI =================
st.set_page_config(layout="wide")
st.title("🔥 V37 PRO - GARMENT AI ANALYZER")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Upload")
    upload_type = st.radio("Type", ["Image","PDF"])
    file = st.file_uploader("Upload file")
    name = st.text_input("Mã hàng")

    if file and st.button("Lưu"):
        if upload_type == "PDF":
            img, pom_img, pom_data = extract_pdf_advanced(file)
        else:
            img = file.read()
            pom_data = {}

        vec = get_vector(img)

        path = f"{name}.png"
        supabase.storage.from_(BUCKET).upload(path, img, upsert=True)
        url = supabase.storage.from_(BUCKET).get_public_url(path)

        supabase.table("ai_data").upsert({
            "file_name": name,
            "vector": vec.tolist(),
            "image_url": url,
            "pom": pom_data
        }).execute()

        st.success("Saved!")
        st.rerun()

with col2:
    st.subheader("So sánh")

    data = supabase.table("ai_data").select("*").execute().data

    if len(data) > 0:
        names = [d['file_name'] for d in data]
        selected = st.selectbox("Chọn mẫu", names)

        size = st.selectbox("Chọn size", ["XS","S","M","L","XL"])

        base = next(d for d in data if d['file_name']==selected)
        base_vec = np.array(base['vector'])

        results = []
        for d in data:
            vec = np.array(d['vector'])
            sim = cosine_similarity([base_vec],[vec])[0][0]
            results.append((d['file_name'], sim, d))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        for name2, sim, d in results[:5]:
            st.write(f"### {name2} - Similarity: {sim*100:.2f}%")
            st.image(d['image_url'])

            # compare POM
            if 'pom' in base and 'pom' in d:
                st.write("So sánh thông số")
                try:
                    df1 = pd.DataFrame(base['pom'])
                    df2 = pd.DataFrame(d['pom'])

                    if size in df1.columns and size in df2.columns:
                        diff = df1[size].astype(float) - df2[size].astype(float)
                        st.dataframe(pd.DataFrame({
                            "POM": df1.iloc[:,0],
                            "Diff": diff
                        }))
                except:
                    st.write("Không đọc được POM")
