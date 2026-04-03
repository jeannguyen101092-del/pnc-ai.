# ==========================================================
# AI FASHION PRO V9 (FINAL OPTIMIZE - GITHUB + NHANH + NHẸ)
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= CONFIG (THAY THÔNG TIN CỦA BẠN) =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
GH_TOKEN = "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"
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

supabase: Client = create_client(URL, KEY)
st.set_page_config(layout="wide", page_title="AI Fashion Pro V9", page_icon="👔")

# ================= AI ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

# ================= HÀM NÉN ẢNH (GIÚP APP CHẠY CỰC NHANH) =================
def compress_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70, optimize=True)
    return buf.getvalue()

# ================= GITHUB UPLOAD =================
def upload_to_github(img_bytes, filename):
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        # Đảm bảo dùng đuôi .png và đường dẫn chuẩn
        url = f"https://github.com{GH_REPO}/contents/imgs/{clean_name}.png"
        headers = {
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        content = base64.b64encode(img_bytes).decode('utf-8')
        
        # Gửi lệnh đẩy ảnh
        data = {"message": f"up {clean_name}", "content": content, "branch": GH_BRANCH}
        res = requests.put(url, headers=headers, json=data, timeout=15)
        
        if res.status_code in [200, 201]:
            return f"https://githubusercontent.com{GH_REPO}/{GH_BRANCH}/imgs/{clean_name}.png"
        else:
            # HIỆN LỖI THẬT SỰ RA MÀN HÌNH ĐỂ BIẾT TẠI SAO SAI
            st.error(f"GitHub từ chối (Mã {res.status_code}): {res.json().get('message')}")
            return None
    except Exception as e:
        st.error(f"Lỗi kết nối mạng: {e}")
        return None

# ================= TRÍCH XUẤT DỮ LIỆU =================
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

VALID_KEYS = ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN','CHEST','LENGTH','SLEEVE','SHOULDER']
BLOCK_KEYS = ['SIZE','SEASON','TECH','DATE','#','FABRIC','BODY','COLOR','WASH']

def extract_specs(table):
    specs = {}
    for r in table:
        if not r: continue
        txt = " | ".join([str(x) for x in r if x]).upper()
        if any(x in txt for x in BLOCK_KEYS): continue
        if not any(k in txt for k in VALID_KEYS): continue
        vals = [parse_val(x) for x in r if x]
        vals = [v for v in vals if v > 0]
        if not vals: continue
        specs[txt[:120]] = round(float(np.median(vals)),2)
    return specs

def classify(specs, text, name):
    txt = (text + name).upper()
    # Tìm Inseam trong mảng specs
    inseam = 0
    for k, v in specs.items():
        if 'INSEAM' in k:
            inseam = v
            break
            
    if 'CARGO' in txt: return "QUẦN CARGO"
    if 'ELASTIC' in txt: return "QUẦN LƯNG THUN"
    if 0 < inseam <= 11: return "QUẦN SHORT"
    if inseam >= 25: return "QUẦN DÀI"
    if 'DRESS' in txt: return "ĐẦM"
    if 'SKIRT' in txt: return "VÁY"
    if 'SHIRT' in txt: return "ÁO SƠ MI"
    return "ÁO"

def get_data(pdf_path):
    try:
        specs, text = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text += t
                for tb in p.extract_tables():
                    specs.update(extract_specs(tb))
        if len(specs) < 3: return None
        doc = fitz.open(pdf_path)
        img_bytes = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
        return {"spec": specs, "img": img_bytes, "cat": classify(specs, text, os.path.basename(pdf_path))}
    except: return None

# ================= SIDEBAR: NẠP KHO & TIẾN ĐỘ =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    
    # Hiển thị số lượng tổng trong kho
    try:
        count_res = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{count_res.count if count_res.count else 0} mẫu")
    except: st.metric("Tổng mẫu trong kho", "0 mẫu")

    st.divider()

    if "up_key" not in st.session_state: st.session_state.up_key = 0
    files = st.file_uploader("Upload PDF nạp kho", accept_multiple_files=True, key=f"up_{st.session_state.up_key}")

    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        p_bar = st.progress(0)
        p_text = st.empty()
        
        for idx, f in enumerate(files):
            try:
                # Cập nhật thanh % tiến độ
                percent = (idx + 1) / len(files)
                p_bar.progress(percent)
                p_text.text(f"Đang nạp: {f.name} ({idx+1}/{len(files)})")

                name = re.sub(r'\s*\(\d+\)', '', f.name)
                with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
                d = get_data("tmp.pdf")
                if not d: continue

                img_small = compress_image(d['img'])
                img_url = upload_to_github(img_small, name)

                tf = transforms.Compose([
                    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])

                with torch.no_grad():
                    vec = ai_brain(tf(Image.open(io.BytesIO(img_small))).unsqueeze(0)).flatten().numpy().tolist()

                supabase.table("ai_data").upsert({
                    "file_name": name, "vector": vec, "spec_json": d['spec'],
                    "img_url": img_url, "category": d['cat']
                }, on_conflict="file_name").execute()
                os.remove("tmp.pdf")
            except Exception as e:
                st.warning(f"Lỗi {f.name}: {e}")

        st.session_state.up_key += 1 # Tự động xóa danh sách file đã đưa lên
        st.success("✅ Nạp kho hoàn tất!")
        st.rerun()

# ================= PHẦN SO SÁNH =================
st.title("👔 AI Fashion Pro V9")
file = st.file_uploader("Tải file đối chứng (Test)", type="pdf")

if file:
    with open("test.pdf","wb") as f: f.write(file.getbuffer())
    target = get_data("test.pdf")

    if target:
        st.info(f"Nhận diện chủng loại: **{target['cat']}**")
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()

        if db.data:
            tf = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img']))).unsqueeze(0)).flatten().numpy()

            results=[]
            for i in db.data:
                if i.get('vector'):
                                results=[]
            for i in db.data:
                if i.get('vector'):
                    # Tính toán similarity chuẩn xác
                    v_t_2d = v_test.reshape(1, -1)
                    v_db_2d = np.array(i['vector']).reshape(1, -1)
                    sim = float(cosine_similarity(v_t_2d, v_db_2d)[0][0]) * 100
                    
                    results.append({
                        "name": i['file_name'],
                        "sim": sim,
                        "spec": i['spec_json'],
                        "img": i.get('img_url') # Lấy link từ GitHub
                    })


            # Hiển thị Top 10
            for r in sorted(results, key=lambda x:x['sim'], reverse=True)[:10]:
                # Đưa % lên trước để không bị che khuất
                with st.expander(f"🎯 {r['sim']:.1f}% | {r['name']}"):
                    c1, c2 = st.columns(2)
                    with c1: st.image(target['img'], caption="Mẫu Test", use_container_width=True)
                    with c2:
                        if r['img']: st.image(r['img'], caption="Mẫu Kho", use_container_width=True)
                        else: st.warning("Không tìm thấy ảnh trên GitHub")

                    # Bảng so sánh
                    diff_data = []
                    poms = set(target['spec']) | set(r['spec'])
                    for p in poms:
                        v1, v2 = target['spec'].get(p, 0), r['spec'].get(p, 0)
                        diff_data.append({"Thông số (POM)": p, "Mẫu Test": v1, "Mẫu Kho": v2, "Chênh lệch": round(v1-v2, 2)})
                    
                    df_res = pd.DataFrame(diff_data)
                    st.dataframe(df_res, use_container_width=True)

                    # Nút xuất Excel
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        df_res.to_excel(writer, index=False)
                    st.download_button(label="📥 Tải Excel So Sánh", data=out.getvalue(), file_name=f"SoSanh_{r['name']}.xlsx")
        else:
            st.warning(f"Chưa có mẫu nào thuộc loại {target['cat']} trong kho.")
