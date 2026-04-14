import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ================= 1. CẤU HÌNH HỆ THỐNG =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
BUCKET = "fashion-imgs"
supabase = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Auditor V96", page_icon="📏")

if 'up_key' not in st.session_state: 
    st.session_state.up_key = 0

st.markdown("""
<style>
.stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
.status-khop { color: #28a745; font-weight: bold; }
.status-lech { color: #dc3545; font-weight: bold; }
thead th { background-color: #f0f2f6 !important; }
</style>
""", unsafe_allow_html=True)

# ================= 2. MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

model_ai = load_model()

def detect_category(text, filename=""):
    t = (str(text) + " " + str(filename)).upper()
    keywords = {
        "VÁY/ĐẦM": ["DRESS", "SKIRT", "VÁY", "ĐẦM", "GOWN"],
        "QUẦN": ["PANT", "JEAN", "SHORT", "TROUSER", "BOTTOM", "QUẦN"],
        "ÁO": ["SHIRT", "JACKET", "HOODIE", "TOP", "TEE", "COAT", "ÁO", "SWEATER"]
    }
    scores = {k:0 for k in keywords}
    for cat, keys in keywords.items():
        for k in keys:
            scores[cat] += t.count(k)
    detected = max(scores, key=scores.get)
    return detected if scores[detected] > 0 else "KHÁC"

def extract_customer(text, filename=""):
    t = (text + " " + filename).upper()
    patterns = [
        r'CUSTOMER[:\s]+([A-Z0-9 _-]+)',
        r'CLIENT[:\s]+([A-Z0-9 _-]+)',
        r'BRAND[:\s]+([A-Z0-9 _-]+)'
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1).strip()
    return "UNKNOWN"

def extract_size(df):
    for col in df.columns:
        for val in df[col].head(5):
            txt = str(val).upper()
            if re.search(r'\b(XXS|XS|S|M|L|XL|XXL|\d{1,2})\b', txt):
                return txt.strip()
    return "UNKNOWN"

def parse_val(t):
    try:
        txt = str(t).replace(',', '.').strip().lower()
        if len(txt) > 10 or not txt: return 0
        match = re.findall(r'(\d+\.\d+|\d+)', txt)
        if not match: return 0
        val = float(match[0])
        if val > 200: return 0
        return val
    except: return 0

def get_image_vector(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        vec = model_ai(tf(img).unsqueeze(0)).flatten().cpu().numpy()
    return vec.astype(float).tolist()

# ================= 3. EXTRACT =================
def extract_pdf_v96(file):
    specs, img_bytes, full_text_list = {}, None, []
    try:
        file.seek(0)
        pdf_content = file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        for page in doc:
            full_text_list.append(str(page.get_text() or ""))
        doc.close()

        full_text = " ".join(full_text_list)

        category = detect_category(full_text, file.name)
        customer = extract_customer(full_text, file.name)

        size_detected = "UNKNOWN"

        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for tb in tables:
                    df = pd.DataFrame(tb).fillna("")
                    
                    if size_detected == "UNKNOWN":
                        size_detected = extract_size(df)

                    flat_text = " ".join(df.astype(str).values.flatten()).upper()
                    if sum(1 for k in ["WAIST","CHEST","HIP","LENGTH"] if k in flat_text) < 2:
                        continue

                    n_col, v_col = -1, -1

                    for r_idx, row in df.head(10).iterrows():
                        for i, v in enumerate(row):
                            if "POM" in str(v).upper():
                                n_col = i
                                break
                        if n_col != -1:
                            break

                    if n_col != -1:
                        max_nums = 0
                        for i in range(len(df.columns)):
                            if i == n_col: continue
                            num_count = sum(1 for val in df.iloc[:12, i] if parse_val(val) > 0)
                            if num_count > max_nums:
                                max_nums = num_count
                                v_col = i

                    if n_col != -1 and v_col != -1:
                        for i in range(len(df)):
                            name = str(df.iloc[i, n_col]).upper().strip()
                            val = parse_val(df.iloc[i, v_col])
                            if len(name) > 3 and val > 0:
                                specs[name] = val
                if specs:
                    break

        return {
            "specs": specs,
            "img": img_bytes,
            "category": category,
            "customer": customer,
            "size": size_detected
        }
    except:
        return None

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.header("📂 KHO MẪU")

    try:
        res_db = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu", res_db.count or 0)
    except:
        st.error("DB lỗi")

    new_files = st.file_uploader("Upload", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")

    if new_files and st.button("NẠP"):
        for f in new_files:
            data = extract_pdf_v96(f)
            if data and data['specs']:
                vec = get_image_vector(data['img'])
                clean = re.sub(r'[^a-zA-Z0-9]', '_', f.name)
                path = f"{clean}.png"

                supabase.storage.from_(BUCKET).upload(path, data['img'], {"upsert":"true"})
                url = supabase.storage.from_(BUCKET).get_public_url(path)

                supabase.table("ai_data").insert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": data['specs'],
                    "image_url": url,
                    "category": data['category'],
                    "customer": data['customer'],
                    "size": data['size']
                }).execute()

        st.success("OK")
        st.session_state.up_key += 1
        st.rerun()

# ================= 5. AUDIT =================
st.title("🔍 AI AUDITOR V96")

file_audit = st.file_uploader("Upload file audit", type="pdf")

customer_filter = st.text_input("Filter khách hàng (optional)")

if file_audit:
    target = extract_pdf_v96(file_audit)

    if target:
        st.info(f"{target['category']} | {target['customer']} | SIZE: {target['size']}")

        query = supabase.table("ai_data").select("*").eq("category", target['category'])

        if customer_filter:
            query = query.ilike("customer", f"%{customer_filter.upper()}%")
        else:
            query = query

        res = query.execute()

        data_pool = res.data

        if not data_pool:
            st.warning("Không có data")
        else:
            target_vec = np.array(get_image_vector(target['img'])).reshape(1, -1)

            matches = []
            for item in data_pool:

                # Ưu tiên đúng size
                size_score = 1 if item.get("size") == target['size'] else 0.7

                sim = cosine_similarity(
                    target_vec,
                    np.array(item['vector']).reshape(1, -1)
                )[0][0]

                final_score = sim * size_score

                matches.append({**item, "sim": final_score})

            top3 = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]

            cols = st.columns(3)
            for i, m in enumerate(top3):
                with cols[i]:
                    st.image(m['image_url'], caption=f"{m['file_name']} ({m['sim']:.2f})")

            selected = st.selectbox("Chọn mẫu", [m['file_name'] for m in top3])
            best = next(m for m in top3 if m['file_name'] == selected)

            rows = []
            for pom, val in target['specs'].items():
                master = best['spec_json'].get(pom, 0)
                diff = round(val - master, 3) if master else 0

                if master == 0:
                    status = "MISS"
                elif abs(diff) > 0.125:
                    status = f"LECH {diff}"
                else:
                    status = "OK"

                rows.append({
                    "POM": pom,
                    "NEW": val,
                    "OLD": master,
                    "KQ": status
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)

            st.download_button(
                "DOWNLOAD EXCEL",
                data=output.getvalue(),
                file_name="audit.xlsx"
            )
