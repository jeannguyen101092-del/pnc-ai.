import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, json
from PIL import Image
from torchvision import models, transforms
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. CẤU HÌNH =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="AI FASHION AUDITOR V33", page_icon="📊")

# ================= 2. KẾT NỐI DB =================
@st.cache_resource
def init_supabase():
    return create_client(URL, KEY)

try:
    supabase: Client = init_supabase()
except Exception as e:
    st.error(f"Lỗi kết nối Supabase: {e}")
    st.stop()

# ================= 3. AI MODEL =================
@st.cache_resource
def load_vision_ai():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_vision_ai()

def get_vector(img_bytes):
    if not img_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            vec = ai_brain(tf(img).unsqueeze(0)).flatten().numpy()
            return vec.tolist()
    except Exception as e:
        return None

# ================= 4. EXCEL =================
def to_excel(df1, df2):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Chi_Tiet', index=False)
        df2.to_excel(writer, sheet_name='POM', index=False)
    return output.getvalue()

# ================= 5. EXTRACT PDF =================
def deep_detail_inspection(text):
    t = str(text).upper()
    det = {}

    if any(x in t for x in ['JACKET', 'VEST', 'COAT']):
        det['Loại'] = "Áo Khoác"
    elif 'DRESS' in t:
        det['Loại'] = "Đầm"
    elif 'PANT' in t:
        det['Loại'] = "Quần"
    else:
        det['Loại'] = "Khác"

    det['Phụ liệu'] = "Zipper" if 'ZIPPER' in t else "Nút/Chun"
    return det

def extract_techpack(pdf_file):
    specs, img, raw_text = {}, None, ""
    
    try:
        pdf_bytes = pdf_file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        doc.close()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text += (page.extract_text() or "")
                tables = page.extract_tables() or []

                for tb in tables:
                    header = [str(h).upper() for h in tb[0]] if tb else []

                    # 🔥 tìm vị trí cột Description
                    desc_idx = -1
                    for i, h in enumerate(header):
                        if "DESCRIPTION" in h:
                            desc_idx = i

                    # nếu không có description thì bỏ
                    if desc_idx == -1:
                        continue

                    for row in tb[1:]:
                        if not row or len(row) <= desc_idx:
                            continue

                        desc = str(row[desc_idx]).upper()

                        # lấy số đo ở các cột sau
                        val_text = " ".join([str(x) for x in row if x])
                        val = re.findall(r"\d+\.?\d*", val_text)

                        if val:
                            specs[desc] = val[0]

        return {
            "spec": specs,
            "img": img,
            "details": deep_detail_inspection(raw_text)
        }

    except:
        return None

# ================= 6. LOAD DATA =================
try:
    res = supabase.table("ai_data").select("*").execute()
    samples = res.data if res.data else []
except:
    samples = []

# ================= 7. SIDEBAR =================
with st.sidebar:
    st.header("Kho dữ liệu")
    st.metric("Số mẫu", len(samples))

    up_pdfs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)

    if up_pdfs and st.button("Nạp dữ liệu"):
        for f in up_pdfs:
            d = extract_techpack(f)
            if not d:
                continue

            ma = f.name.replace(".pdf", "")

            # upload ảnh
            img_path = f"{ma}.png"
            img_url = None
            try:
                supabase.storage.from_(BUCKET).upload(
                    img_path,
                    d['img'],
                    file_options={"content-type": "image/png"},
                    upsert=True
                )
                img_url = supabase.storage.from_(BUCKET).get_public_url(img_path)
            except:
                pass

            vec = get_vector(d['img'])

            supabase.table("ai_data").upsert({
                "file_name": ma,
                "vector": vec,
                "spec_json": d['spec'],
                "details": d['details'],
                "image_url": img_url
            }).execute()

        st.success("Đã upload xong")
        st.rerun()

# ================= 8. MAIN =================
st.title("AI FASHION AUDITOR")

sample_list = ["AUTO AI"] + [s['file_name'] for s in samples]
selected_code = st.selectbox("Chọn mã gốc", sample_list)

test_files = st.file_uploader("Upload file test", type="pdf", accept_multiple_files=True)

if test_files:
    for t_file in test_files:
        data_test = extract_techpack(t_file)
        if not data_test:
            continue

        with st.expander(t_file.name, expanded=True):

            best_match = None
            sim_score = 0

            v_test = get_vector(data_test['img'])

            if v_test:
                vt = np.array(v_test).reshape(1, -1)

                for s in samples:
                    try:
                        v_raw = s.get('vector')

                        if isinstance(v_raw, str):
                            v_raw = json.loads(v_raw)

                        if not v_raw:
                            continue

                        vs = np.array(v_raw).reshape(1, -1)

                        score = cosine_similarity(vt, vs)[0][0]

                        if score > sim_score:
                            sim_score = score
                            best_match = s

                    except:
                        continue

            if best_match:
                st.success(f"Match: {best_match['file_name']} | {sim_score:.2%}")

                g_spec = best_match.get('spec_json', {})
                if isinstance(g_spec, str):
                    g_spec = json.loads(g_spec)

                df_pom = pd.DataFrame([
                    {
                        "Key": k,
                        "Test": data_test['spec'].get(k, "-"),
                        "Gốc": g_spec.get(k, "-"),
                        "OK": data_test['spec'].get(k) == g_spec.get(k)
                    }
                    for k in set(list(data_test['spec']) + list(g_spec))
                ])

                st.dataframe(df_pom)

                st.download_button(
                    "Download Excel",
                    to_excel(df_pom, df_pom),
                    file_name="compare.xlsx"
                )

            else:
                st.warning("Không tìm thấy mẫu")
