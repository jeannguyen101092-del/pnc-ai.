import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd, numpy as np
import torch, random, datetime, time
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# ================= CONFIG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

try:
    supabase: Client = create_client(URL, KEY)
    supabase.table("ai_data").select("file_name").limit(1).execute()
except Exception as e:
    st.error(f"❌ Lỗi kết nối Supabase: {e}")

st.set_page_config(layout="wide", page_title="AI FASHION PRO V16", page_icon="🛡️")

if "target" not in st.session_state: st.session_state.target = None
if "up_key" not in st.session_state: st.session_state.up_key = 200

@st.cache_resource
def load_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= FUNCTIONS =================

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
            return ai_brain(tf(img).unsqueeze(0)).flatten().numpy()
    except Exception as e:
        st.error(f"Vector lỗi: {e}")
        return None


def parse_val(t):
    try:
        t_str = str(t).strip()
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', t_str)
        if not found: return 0
        v = found[0]
        if " " in v:
            a, b = v.split()
            return float(a) + eval(b)
        if "/" in v: return eval(v)
        return float(v)
    except:
        return 0


def excel_to_img(file_obj):
    try:
        df = pd.read_excel(file_obj, engine="openpyxl").dropna(how='all').fillna("")
        fig, ax = plt.subplots(figsize=(22, len(df.head(60)) * 0.6 + 2))
        ax.axis('off')
        ax.table(cellText=df.head(60).values, colLabels=df.columns, loc='center').scale(1.2, 2.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        st.error(f"❌ Excel lỗi: {e}")
        return None


def get_data(pdf_path):
    specs, text = {}, ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text += t
                for tb in p.extract_tables():
                    if not tb: continue
                    for r in tb:
                        if not r or len(r) < 2: continue
                        val = parse_val(r[-1])
                        if val > 0.1:
                            desc = " ".join([str(x or "") for x in r[:-1]]).upper()
                            specs[desc[:120]] = val

        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()
        return {"spec": specs, "img": img, "text": text}
    except Exception as e:
        st.error(f"❌ PDF lỗi: {e}")
        return None

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📦 KHO AI V16")

    try:
        samples = supabase.table("ai_data").select("*").execute().data
    except Exception as e:
        st.error(f"DB lỗi: {e}")
        samples = []

    st.metric("Tổng mẫu", len(samples))

    files = st.file_uploader("Upload PDF + Excel", accept_multiple_files=True,
                             type=['pdf', 'xlsx', 'xls'], key=f"up_{st.session_state.up_key}")

    if files and st.button("🚀 NẠP KHO"):
        groups = {}
        for f in files:
            nums = re.findall(r'\d{3,}', f.name)
            if nums:
                ma = max(nums, key=len)
                ext = "." + f.name.split('.')[-1].lower()
                groups.setdefault(ma, {})[ext] = f

        for ma, p in groups.items():
            f_pdf = p.get('.pdf')
            f_exl = p.get('.xlsx') or p.get('.xls')

            if not f_pdf or not f_exl:
                st.warning(f"⚠️ Thiếu file {ma}")
                continue

            with st.spinner(f"Đang xử lý {ma}..."):
                open("tmp.pdf", "wb").write(f_pdf.getbuffer())

                d = get_data("tmp.pdf")
                ex_img = excel_to_img(f_exl)

                if not d or not ex_img:
                    continue

                vec_raw = get_vector(d['img'])
                if vec_raw is None:
                    continue

                vec = vec_raw.tolist()

                try:
                    supabase.storage.from_(BUCKET).upload(f"{ma}_t.webp", d['img'], {"x-upsert": "true"})
                    supabase.storage.from_(BUCKET).upload(f"{ma}_e.webp", ex_img, {"x-upsert": "true"})
                except Exception as e:
                    st.error(f"Upload lỗi: {e}")
                    continue

                try:
                    u_t = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_t.webp")
                    u_e = supabase.storage.from_(BUCKET).get_public_url(f"{ma}_e.webp")

                    supabase.table("ai_data").upsert({
                        "file_name": ma,
                        "vector": vec,
                        "spec_json": d['spec'],
                        "img_url": u_t,
                        "excel_img_url": u_e
                    }).execute()

                    st.success(f"✅ Xong {ma}")

                except Exception as e:
                    st.error(f"DB lỗi: {e}")

        st.session_state.up_key += 1
        st.rerun()

# ================= MAIN =================
st.title("🛡️ AI FASHION PRO V16")

st.info("✔️ V16 đã fix: upload, excel, vector, supabase, debug")

# ===== SO SÁNH + XUẤT EXCEL =====
test_pdf = st.file_uploader("📄 Upload PDF TEST để so sánh", type="pdf")

if test_pdf:
    open("test.pdf", "wb").write(test_pdf.getbuffer())
    data_test = get_data("test.pdf")

    if data_test:
        try:
            samples = supabase.table("ai_data").select("*").execute().data
        except:
            samples = []

        if not samples:
            st.warning("⚠️ Kho chưa có dữ liệu")
        else:
            if st.button("🤖 Tự động tìm mẫu giống nhất"):
                test_vec = get_vector(data_test['img'])
                best_sim, best_s = -1, None

                for s in samples:
                    if s.get('vector'):
                        sim = cosine_similarity([test_vec], [np.array(s['vector'])])[0][0]
                        if sim > best_sim:
                            best_sim = sim
                            best_s = s

                st.session_state.target = best_s
                st.rerun()

        target = st.session_state.target

        if target:
            st.subheader(f"📊 So sánh với: {target['file_name']}")

            rows = []
            for kt, vt in data_test['spec'].items():
                best_m, high_r = None, 0

                for kb in target['spec_json'].keys():
                    r = SequenceMatcher(None, kt, kb).ratio()
                    if r > high_r:
                        high_r, best_m = r, kb

                v_db = target['spec_json'].get(best_m, 0) if high_r > 0.6 else "N/A"
                diff = round(vt - v_db, 3) if isinstance(v_db, (int, float)) else "N/A"

                rows.append({
                    "Thông số": kt,
                    "Test": vt,
                    "Kho": v_db,
                    "Chênh lệch": diff,
                    "Trạng thái": "OK" if diff == 0 else "SAI"
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=500)

            # ===== NÚT XUẤT EXCEL =====
            output = io.BytesIO()
            df.to_excel(output, index=False)

            st.download_button(
                "📥 XUẤT FILE EXCEL SO SÁNH",
                data=output.getvalue(),
                file_name=f"compare_{target['file_name']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("👉 Hãy chọn hoặc AI nhận diện mẫu để so sánh")
