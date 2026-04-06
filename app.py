import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, datetime, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Lỗi kết nối Database: {e}")

st.set_page_config(layout="wide", page_title="AI FASHION PRO V17 PRO", page_icon="🛡️")

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 1500

# ================= AI =================
@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

# ================= CORE =================
def get_vector(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy().tolist()
    except Exception as e:
        st.error(f"Vector lỗi: {e}")
        return None

# 🔥 OCR fallback (V17 mới)
def ocr_fallback(img_bytes):
    try:
        import pytesseract
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except:
        return ""

# 🔥 PDF đọc mạnh hơn
def extract_pdf_ultimate(pdf_path):
    specs, text = {}, ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text += txt
                for tb in page.extract_tables():
                    if not tb: continue
                    for r in tb:
                        if len(r) >= 2:
                            k = str(r[0]).upper()
                            v = str(r[-1])
                            if len(k) > 3 and v:
                                specs[k[:100]] = v

        # 🔥 nếu không đọc được bảng → dùng OCR
        if len(specs) == 0:
            doc = fitz.open(pdf_path)
            img = doc.load_page(0).get_pixmap().tobytes("png")
            text = ocr_fallback(img)
            specs["OCR_TEXT"] = text[:500]
            doc.close()

        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2,2)).tobytes("png")
        doc.close()

        return {"spec": specs, "img": img, "text": text}
    except Exception as e:
        st.error(f"PDF lỗi: {e}")
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 KHO V17 PRO")

    try:
        samples = supabase.table("ai_data").select("*").execute().data
    except:
        samples = []

    st.metric("Tổng mẫu", len(samples))

    files = st.file_uploader("Upload PDF + Excel", accept_multiple_files=True,
                             type=['pdf','xlsx','xls'], key=st.session_state.up_key)

    if files and st.button("🚀 NẠP KHO AI"):
        for f in files:
            if f.name.endswith('.pdf'):
                ma = re.findall(r'\d{3,}', f.name)
                if not ma: continue
                ma = ma[0]

                open("tmp.pdf","wb").write(f.getbuffer())
                d = extract_pdf_ultimate("tmp.pdf")

                if not d: continue

                vec = get_vector(d['img'])
                if not vec: continue

                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}.png", d['img'], {"x-upsert":"true"})
                    url = supabase.storage.from_(BUCKET).get_public_url(f"{ma}.png")

                    supabase.table("ai_data").upsert({
                        "file_name": ma,
                        "vector": vec,
                        "spec_json": d['spec'],
                        "img_url": url
                    }).execute()

                    st.success(f"✅ {ma}")
                except Exception as e:
                    st.error(e)

        st.session_state.up_key += 1
        st.rerun()

# ================= MAIN =================
st.title("🛡️ AI FASHION PRO V17 PRO MAX")

pdf_test = st.file_uploader("📄 Upload PDF TEST", type="pdf")

if pdf_test:
    open("test.pdf","wb").write(pdf_test.getbuffer())
    d_test = extract_pdf_ultimate("test.pdf")

    if d_test:
        st.image(d_test['img'])

        if st.button("🤖 AI MATCH"):
            vec_test = get_vector(d_test['img'])

            best, score_max = None, 0
            for s in samples:
                if s.get('vector'):
                    sc = cosine_similarity([vec_test],[s['vector']])[0][0]
                    if sc > score_max:
                        score_max = sc
                        best = s

            st.session_state.target = best
            st.rerun()

        target = st.session_state.target

        if target:
            st.image(target['img_url'])

            rows = []
            for k,v in d_test['spec'].items():
                v2 = target['spec_json'].get(k,"-")
                rows.append([k,v,v2])

            df = pd.DataFrame(rows, columns=["Thông số","Test","Kho"])
            st.dataframe(df)

            # 🔥 EXPORT EXCEL CHUẨN
            out = io.BytesIO()
            df.to_excel(out,index=False)

            st.download_button(
                "📥 XUẤT EXCEL XỊN",
                out.getvalue(),
                file_name="compare.xlsx"
            )

else:
    st.info("Upload PDF để bắt đầu")
