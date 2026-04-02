import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. CẤU HÌNH (Thay URL và KEY của bạn vào đây)
# ==========================================
URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI MASTER PRO CLOUD", page_icon="📊")

# Khởi tạo bộ nhớ tạm cho phiên làm việc
if 'sel_code' not in st.session_state: st.session_state.sel_code = None

# --- HÀM HỖ TRỢ AI ---
@st.cache_resource
def load_ai():
    model = models.resnet18(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()
ai_brain = load_ai()

def clean_pom_name(text):
    if not text: return ""
    t = str(text).strip().upper()
    t = re.sub(r'[-\d\s/]{3,}', '', t)
    return t.strip()

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

def get_data(pdf_path):
    specs, all_texts = {}, []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt: all_texts.append(txt)
                for table in p.extract_tables():
                    if len(table) < 2: continue
                    for r in table:
                        if not r or len(r) < 2: continue
                        val = parse_val(r[-1])
                        pom_n = clean_pom_name(" ".join([str(x) for x in r[:-1] if x]))
                        if val > 0 and len(pom_n) > 3: specs[pom_n] = val
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap()
        cat = "ÁO" if any(x in str(all_texts).upper() for x in ['CHEST', 'BUST', 'ARMHOLE']) else "QUẦN"
        return {"spec": specs, "img_bytes": pix.tobytes("png"), "cat": cat}
    except: return None

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ AI MASTER PRO CLOUD")

with st.sidebar:
    st.header("⚙️ QUẢN LÝ KHO")
    try:
        res_count = supabase.table("ai_data").select("file_name", count="exact").execute()
        count = res_count.count if res_count.count else 0
        st.metric("📁 Tổng file trong kho", count)
    except:
        st.error("Lỗi kết nối bảng 'ai_data' trên Supabase!")

up = st.file_uploader("📥 Upload file PDF mới để phân tích/so sánh", type="pdf")

if up:
    with open("temp.pdf", "wb") as f: f.write(up.getbuffer())
    target = get_data("temp.pdf")
    
    if target:
        # Tính Vector AI cho file vừa upload
        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            img_input = Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')
            target_v = ai_brain(tf(img_input).unsqueeze(0)).flatten().numpy()

        # 1. KIỂM TRA KHO VÀ SO SÁNH
        db_res = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        
        if db_res.data:
            res_list = []
            for item in db_res.data:
                sim = float(cosine_similarity([target_v], [np.array(item['vector'])])) * 100
                res_list.append({"name": item['file_name'], "sim": sim, "spec": item['spec_json']})
            
            res_list = sorted(res_list, key=lambda x: x['sim'], reverse=True)[:4]

            # Tự động chọn mã giống nhất
            if st.session_state.sel_code is None: st.session_state.sel_code = res_list[0]['name']

            st.subheader(f"🤖 GỢI Ý MÃ {target['cat']} TƯƠNG ĐỒNG TRONG KHO:")
            cols = st.columns(4)
            for i, item in enumerate(res_list):
                with cols[i]:
                    st.info(f"📄 {item['name']}")
                    st.write(f"Khớp: **{item['sim']:.1f}%**")
                    if st.button("CHỌN MÃ NÀY", key=item['name']):
                        st.session_state.sel_code = item['name']
                        st.rerun()

            # BẢNG SO SÁNH
            st.divider()
            ref = next(x for x in res_list if x['name'] == st.session_state.sel_code)
            st.subheader(f"📊 SO SÁNH: FILE MỚI vs {st.session_state.sel_code}")
            
            diff_data = []
            all_poms = sorted(list(set(target['spec'].keys()) | set(ref['spec'].keys())))
            for p in all_poms:
                v1, v2 = target['spec'].get(p, 0), ref['spec'].get(p, 0)
                diff_data.append({"Thông số (POM)": p, "Mẫu Mới": v1, "Mẫu Kho": v2, "Chênh lệch": round(v1-v2, 3)})
            
            df = pd.DataFrame(diff_data)
            st.table(df)

            # Nút xuất Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 TẢI EXCEL SO SÁNH", output.getvalue(), f"So_sanh_{up.name}.xlsx")
        else:
            st.warning("⚠️ Kho hiện đang trống hoặc không có mẫu cùng loại (Áo/Quần).")

        # 2. NÚT LƯU VÀO KHO (Để xây dựng kho dữ liệu)
        st.divider()
        st.subheader("📥 QUẢN LÝ DỮ LIỆU")
        if st.button("➕ LƯU FILE NÀY VÀO KHO DỮ LIỆU"):
            payload = {
                "file_name": up.name,
                "vector": target_v.tolist(),
                "spec_json": target['spec'],
                "category": target['cat']
            }
            try:
                supabase.table("ai_data").upsert(payload).execute()
                st.success(f"✅ Đã lưu {up.name} vào hệ thống thành công!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi lưu: {e}")
