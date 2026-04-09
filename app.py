# ==========================================================
# AI FASHION PRO V37.7 - FIXED VECTOR & ERROR HIGHLIGHT
# ==========================================================

import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- KẾT NỐI (Thay URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V37.7", page_icon="📊")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- CÔNG CỤ ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.')
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

def extract_specs(pdf_file):
    specs, img_data = {}, None
    pdf_bytes = pdf_file.read()
    
    # 1. Ảnh trang đầu
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img_data = pix.tobytes("png")
    doc.close()

    # 2. Quét POM (Chỉ trang có từ khóa)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = (page.extract_text() or "").upper()
            if any(k in txt for k in ["POM", "SPEC", "MEASURE"]):
                tables = page.extract_tables()
                for tb in tables:
                    for r in tb:
                        if not r or len(r) < 2: continue
                        desc = str(r[0]).replace('\n',' ').strip().upper()
                        if len(desc) < 3 or any(x in desc for x in ['DATE', 'PAGE']): continue
                        vals = [parse_val(x) for x in r[1:] if x]
                        vals = [v for v in vals if v > 0]
                        if vals: specs[desc] = round(float(np.median(vals)), 2)
    return {"spec": specs, "img": img_data, "text": txt}

# --- STYLE BẢNG (Bôi đỏ dòng lệch) ---
def highlight_diff(val):
    color = 'red' if abs(val) > 0.5 else 'black'
    return f'color: {color}'

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 NẠP KHO")
    files = st.file_uploader("Upload Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        p_bar = st.progress(0)
        for i, f in enumerate(files):
            d = extract_specs(f)
            if d['spec']:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                with torch.no_grad():
                    vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                    "img_base64": base64.b64encode(d['img']).decode(),
                    "category": "QUẦN" if "INSEAM" in str(d['spec']) else "ÁO"
                }).execute()
            p_bar.progress((i + 1) / len(files))
        st.success("✅ Đã nạp!")

# --- MAIN ---
st.title("🔍 AI Fashion Pro V37.7")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf")

if test_file:
    target = extract_specs(test_file)
    if target['spec']:
        st.info(f"Phát hiện {len(target['spec'])} thông số POM")
        
        # Load DB
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            with torch.no_grad():
                v_test = ai_brain(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                # Ép kiểu vector về numpy array để tránh lỗi TypeError
                v_ref = np.array(i['vector']).reshape(1, -1)
                sim = float(cosine_similarity(v_test, v_ref)[0][0]) * 100
                matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[:3]

            for m in best:
                with st.expander(f"Khớp: {m['name']} ({m['sim']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img'], caption="File kiểm")
                    c2.image(base64.b64decode(m['img']), caption="Mẫu gốc")

                    diff = []
                    for p, v1 in target['spec'].items():
                        v2 = m['spec'].get(p, 0)
                        if v2 == 0:
                            v2 = next((val for k, val in m['spec'].items() if p[:8] in k), 0)
                        diff.append({"Hạng mục": p, "Thực tế": v1, "Mẫu gốc": v2, "Lệch": round(v1-v2, 2)})
                    
                    df = pd.DataFrame(diff)
                    # Áp dụng bôi đỏ dòng lệch > 0.5
                    st.dataframe(df.style.applymap(highlight_diff, subset=['Lệch']))
                    
                    # Nút xuất Excel
                    out = io.BytesIO()
                    df.to_excel(out, index=False)
                    st.download_button(f"📥 Báo cáo {m['name']}", out.getvalue(), f"Audit_{m['name']}.xlsx")
