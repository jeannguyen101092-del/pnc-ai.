# ==========================================================
# AI FASHION PRO V8.3 - STABLE & OPTIMIZED
# ==========================================================

import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, requests
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ================= 1. CẤU HÌNH (THAY KEY CỦA BẠN VÀO ĐÂY) =================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

GH_TOKEN = st.secrets["GH_TOKEN"] 
# CHỈ điền tên_user/tên_kho. KHÔNG điền link website
GH_REPO = "jeannguyen101092-del/fashion-storage"
GH_BRANCH = "main"

# Khởi tạo kết nối
try:
    supabase: Client = create_client(URL, KEY)
except:
    st.error("Lỗi cấu hình Supabase. Vui lòng kiểm tra lại URL/KEY.")

st.set_page_config(layout="wide", page_title="AI Fashion Pro V8.3", page_icon="👔")

# ================= 2. AI & GITHUB ENGINE =================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return torch.nn.Sequential(*(list(model.children())[:-1])).eval()

ai_brain = load_ai()

def upload_to_github(img_bytes, filename):
    """Tải ảnh lên GitHub với cơ chế chống treo app"""
    try:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        url = f"https://github.com{GH_REPO}/contents/imgs/{clean_name}.png"
        headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        content = base64.b64encode(img_bytes).decode()
        
        # Kiểm tra file tồn tại (Timeout 10s tránh đứng máy)
        check = requests.get(url, headers=headers, timeout=10)
        data = {"message": f"Up {filename}", "content": content, "branch": GH_BRANCH}
        if check.status_code == 200:
            data["sha"] = check.json()["sha"]
            
        res = requests.put(url, headers=headers, json=data, timeout=15)
        if res.status_code in [200, 201]:
            return f"https://githubusercontent.com{GH_REPO}/{GH_BRANCH}/imgs/{clean_name}.png"
        return None
    except Exception as e:
        st.warning(f"⚠️ Lỗi kết nối GitHub với file {filename}: {e}")
        return None

# ================= 3. XỬ LÝ DỮ LIỆU PDF =================
def parse_val(t):
    try:
        f = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not f: return 0
        v = f[0]
        return eval(v.replace(' ', '+')) if '/' in v else float(v)
    except: return 0

def get_data(pdf_path):
    try:
        specs, txt = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                txt += (p.extract_text() or "") + " "
                for tbl in p.extract_tables():
                    for r in tbl:
                        if not r or len(r) < 2: continue
                        line = " | ".join([str(x) for x in r if x]).upper()
                        if any(k in line for k in ['INSEAM','WAIST','HIP','THIGH','KNEE','LEG OPEN','CHEST','LENGTH','SLEEVE']):
                            vals = [parse_val(x) for x in r if parse_val(x) > 0]
                            if vals: specs[line[:80]] = round(float(np.median(vals)), 2)
        
        if len(specs) < 5: return {"err": "Thiếu thông số (Dưới 5 dòng)"}
        
        doc = fitz.open(pdf_path)
        img = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).tobytes("png")
        if len(img) < 1000: return {"err": "Không tìm thấy hình ảnh"}

        # Phân loại chủng loại
        cat = "ÁO"
        ins = next((v for k, v in specs.items() if 'INSEAM' in k), 0)
        if ins > 0: cat = "QUẦN SHORT" if ins <= 12 else "QUẦN DÀI"
        elif any(x in txt.upper() for x in ['DRESS', 'ĐẦM', 'SKIRT']): cat = "ĐẦM/VÁY"
        
        return {"spec": specs, "img": img, "cat": cat}
    except Exception as e: return {"err": str(e)}

# ================= 4. GIAO DIỆN SIDEBAR =================
with st.sidebar:
    st.header("📦 QUẢN LÝ KHO")
    try:
        count_res = supabase.table("ai_data").select("*", count="exact").execute()
        st.metric("Tổng mẫu trong kho", f"{count_res.count if count_res.count else 0} mẫu")
    except: st.metric("Tổng mẫu trong kho", "Đang kết nối...")

    st.divider()
    if "up_key" not in st.session_state: st.session_state.up_key = 0
    files = st.file_uploader("Chọn PDF nạp kho", accept_multiple_files=True, key=f"u_{st.session_state.up_key}")

    if files and st.button("🚀 BẮT ĐẦU NẠP"):
        bar = st.progress(0)
        msg = st.empty()
        success = 0
        
        for idx, f in enumerate(files):
            msg.text(f"Đang xử lý: {f.name} ({idx+1}/{len(files)})")
            bar.progress((idx + 1) / len(files))
            
            with open("tmp.pdf", "wb") as t: t.write(f.getbuffer())
            d = get_data("tmp.pdf")
            
            if "err" in d:
                st.warning(f"Bỏ qua {f.name}: {d['err']}")
                continue
            
            url_anh = upload_to_github(d['img'], f.name)
            if not url_anh: continue # Bỏ qua nếu lỗi mạng GitHub

            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                vec = ai_brain(tf(Image.open(io.BytesIO(d['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()

            supabase.table("ai_data").upsert({
                "file_name": f.name, "vector": vec, "spec_json": d['spec'], "img_url": url_anh, "category": d['cat']
            }, on_conflict="file_name").execute()
            
            success += 1
            os.remove("tmp.pdf")

        st.session_state.up_key += 1
        st.success(f"✅ Đã nạp thành công {success} mẫu!")
        st.rerun()

# ================= 5. GIAO DIỆN SO SÁNH =================
st.title("👔 AI Fashion Pro V8.3")
test_file = st.file_uploader("Tải file đối chứng", type="pdf")

if test_file:
    with open("test.pdf", "wb") as f: f.write(test_file.getbuffer())
    target = get_data("test.pdf")

    if "err" in target:
        st.error(f"Lỗi: {target['err']}")
    else:
        st.info(f"Loại sản phẩm nhận diện: **{target['cat']}**")
        db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()

        if db.data:
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            results = []
            for i in db.data:
                if i.get('vector'):
                    sim = float(cosine_similarity([v_test], [np.array(i['vector'])])) * 100
                    results.append({"name": i['file_name'], "sim": sim, "spec": i['spec_json'], "img": i['img_url']})

            for r in sorted(results, key=lambda x: x['sim'], reverse=True)[:5]:
                with st.expander(f"🎯 {r['sim']:.1f}% | {r['name']}"):
                    c1, c2 = st.columns(2)
                    c1.image(target['img'], caption="Mẫu Test")
                    c2.image(r['img'], caption="Mẫu Kho")

                    diff = []
                    poms = set(target['spec']) | set(r['spec'])
                    for p in poms:
                        v1, v2 = target['spec'].get(p, 0), r['spec'].get(p, 0)
                        diff.append({"POM": p, "Test": v1, "Kho": v2, "Lệch": round(v1-v2, 2)})
                    
                    df_res = pd.DataFrame(diff)
                    st.dataframe(df_res, use_container_width=True)

                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as wr: df_res.to_excel(wr, index=False)
                    st.download_button(label="📥 Tải Excel", data=out.getvalue(), file_name=f"SoSanh_{r['name']}.xlsx")
        else:
            st.warning(f"Chưa có mẫu nào thuộc loại {target['cat']} trong kho.")
