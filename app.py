# ==========================================================
# AI FASHION PRO V37.8 - FIXED STYLER & POM SCAN
# ==========================================================
import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, base64, json
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# --- CONFIG (Thay URL/KEY của bạn) ---
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V37.8", page_icon="📊")

@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# --- CÔNG CỤ QUÉT SỐ ---
def parse_val(t):
    try:
        txt = str(t).replace(',', '.').replace('"', '').strip()
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', txt)
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return 0

# --- FIX LỖI BÔI ĐỎ (Dùng map thay cho applymap) ---
def style_diff(val):
    color = 'red' if abs(val) > 0.5 else 'white'
    return f'color: {color}; font-weight: bold'

def extract_specs_deep(pdf_file):
    specs, img_data = {}, None
    pdf_bytes = pdf_file.read()
    
    # 1. Ảnh trang đầu (Sketch)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    img_data = pix.tobytes("png")
    doc.close()

    # 2. QUÉT POM SÂU: Quét cả Table và Text tự do
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # A. Quét bảng truyền thống
            tables = page.extract_tables()
            for tb in tables:
                for r in tb:
                    if not r or len(r) < 2: continue
                    desc = str(r[0]).replace('\n',' ').strip().upper()
                    if len(desc) < 3 or any(x in desc for x in ['DATE','PAGE','FABRIC']): continue
                    vals = [parse_val(x) for x in r[1:] if x]
                    vals = [v for v in vals if v > 0]
                    if vals: specs[desc] = round(float(np.median(vals)), 2)
            
            # B. Nếu bảng rỗng, quét text tự do (Dành cho file Express/Nike)
            if not specs:
                lines = (page.extract_text() or "").split("\n")
                for line in lines:
                    nums = re.findall(r"\d+\.?\d*", line)
                    if len(nums) >= 2 and len(line) > 20:
                        parts = re.split(r"\s{2,}", line.strip())
                        if len(parts) >= 2:
                            specs[parts[0].upper()] = parse_val(nums[0])

    return {"spec": specs, "img": img_data}

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 QUẢN LÝ KHO")
    files = st.file_uploader("Upload Techpacks", accept_multiple_files=True)
    if files and st.button("🚀 NẠP DỮ LIỆU"):
        for f in files:
            d = extract_specs_deep(f)
            if d['spec']:
                tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                img_p = Image.open(io.BytesIO(d['img'])).convert('RGB')
                with torch.no_grad():
                    vec = ai_brain(tf(img_p).unsqueeze(0)).flatten().numpy().tolist()
                supabase.table("ai_data").upsert({
                    "file_name": f.name, "vector": vec, "spec_json": d['spec'],
                    "img_base64": base64.b64encode(d['img']).decode()
                }).execute()
        st.success("✅ Đã nạp xong!")

# --- MAIN ---
st.title("🔍 AI Fashion Pro V37.8")
test_file = st.file_uploader("Upload file kiểm tra", type="pdf")

if test_file:
    target = extract_specs_deep(test_file)
    if target['spec']:
        res = supabase.table("ai_data").select("*").execute()
        if res.data:
            img_t = Image.open(io.BytesIO(target['img'])).convert('RGB')
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(img_t).unsqueeze(0)).flatten().numpy().reshape(1, -1)

            matches = []
            for i in res.data:
                v_ref = np.array(i['vector']).reshape(1, -1)
                sim = float(cosine_similarity(v_test, v_ref)) * 100
                matches.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_base64']})
            
            best = sorted(matches, key=lambda x: x['sim'], reverse=True)[:1]

            for m in best:
                st.subheader(f"✨ Khớp nhất: {m['name']} ({m['sim']:.1f}%)")
                c1, c2 = st.columns(2)
                c1.image(target['img'], caption="File đang kiểm")
                c2.image(base64.b64decode(m['img']), caption="Mẫu gốc")

                # SO SÁNH THÔNG SỐ (TS)
                diff = []
                for p, v1 in target['spec'].items():
                    v2 = m['spec'].get(p, 0)
                    if v2 == 0:
                        v2 = next((val for k, val in m['spec'].items() if p[:6] in k), 0)
                    diff.append({"Hạng mục": p, "Mẫu kiểm": v1, "Mẫu gốc": v2, "Lệch": round(v1-v2, 2)})
                
                df = pd.DataFrame(diff)
                # FIX LỖI BÁO ĐỎ TRONG HÌNH
                st.table(df.style.map(style_diff, subset=['Lệch']))
                
                out = io.BytesIO()
                df.to_excel(out, index=False)
                st.download_button("📥 Xuất Excel", out.getvalue(), f"Report_{m['name']}.xlsx")
    else:
        st.error("⚠️ Không tìm thấy bảng thông số (POM). Hãy kiểm tra lại trang POM trong PDF.")
