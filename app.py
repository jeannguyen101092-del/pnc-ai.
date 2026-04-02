import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. CẤU HÌNH SUPABASE (Thay bằng thông tin từ ảnh của bạn)
# ==========================================
SUPABASE_URL = "https://supabase.co" 
SUPABASE_KEY = "your-anon-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(layout="wide", page_title="AI SMART SPEC V60", page_icon="📊")

# Khởi tạo session state
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
    t = re.sub(r'[:;|+]', '', t)
    return t.strip()

def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return None
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except: return None

def get_data(pdf_path):
    specs, all_texts = {}, []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt: all_texts.append(txt)
                for table in p.extract_tables():
                    if not table or len(table) < 2: continue
                    # (Giữ nguyên logic lấy specs của bạn ở đây...)
                    for r in table:
                        raw_pom = " ".join([str(x) for x in r if x])
                        pom_n = clean_pom_name(raw_pom)
                        val = parse_val(r[-1]) # Giả định cột cuối là giá trị
                        if val and len(pom_n) > 3: specs[pom_n] = val
        doc = fitz.open(pdf_path)
        pix = doc.load_page(0).get_pixmap()
        return {"spec": specs, "img_bytes": pix.tobytes("png"), "cat": "ÁO" if "CHEST" in str(all_texts).upper() else "QUẦN"}
    except: return None

# --- GIAO DIỆN ---
st.title("👖 AI SMART SPEC V60 - SUPABASE")

with st.sidebar:
    st.header("⚙️ QUẢN TRỊ KHO")
    res_count = supabase.table("spec_db").select("file_name", count="exact").execute()
    st.metric("📁 Tổng file trong kho", res_count.count if res_count.count else 0)

up = st.file_uploader("📥 Tải lên Spec PDF mới", type="pdf")

if up:
    with open("temp.pdf", "wb") as f: f.write(up.getbuffer())
    target = get_data("temp.pdf")
    
    if target:
        # Lấy dữ liệu từ Supabase để so sánh
        db_res = supabase.table("spec_db").select("*").eq("category", target['cat']).execute()
        
        if not db_res.data:
            st.error("❌ Không có dữ liệu cùng loại trong kho!")
        else:
            # 1. AI Tìm kiếm (Cosine Similarity)
            tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                target_v = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()
            
            res_list = []
            for item in db_res.data:
                sim = float(cosine_similarity([target_v], [np.array(item['vector'])])[0][0]) * 100
                res_list.append({"name": item['file_name'], "sim": sim, "spec": item['spec_json']})
            
            res_list = sorted(res_list, key=lambda x: x['sim'], reverse=True)[:4]

            # 2. Tự động chọn mã cao nhất nếu chưa chọn thủ công
            if st.session_state.sel_code is None:
                st.session_state.sel_code = res_list[0]['name']

            st.subheader(f"🤖 GỢI Ý MÃ {target['cat']} TƯƠNG ĐỒNG:")
            cols = st.columns(4)
            for idx, item in enumerate(res_list):
                with cols[idx]:
                    st.write(f"📌 {item['name']}")
                    st.caption(f"Độ khớp: {item['sim']:.1f}%")
                    if st.button(f"CHỌN MÃ NÀY", key=item['name']):
                        st.session_state.sel_code = item['name']
                        st.rerun()

            # 3. So sánh chi tiết & Xuất file
            st.divider()
            ref_data = next((x for x in res_list if x['name'] == st.session_state.sel_code), res_list[0])
            
            st.subheader(f"📊 SO SÁNH: FILE MỚI vs {st.session_state.sel_code}")
            
            diff_data = []
            all_poms = sorted(list(set(target['spec'].keys()) | set(ref_data['spec'].keys())))
            for p in all_poms:
                v1 = target['spec'].get(p, 0)
                v2 = ref_data['spec'].get(p, 0)
                diff = v1 - v2
                diff_data.append({"POM": p, "FILE MỚI": v1, "TRONG KHO": v2, "LỆCH": diff})
            
            df = pd.DataFrame(diff_data)
            st.table(df)

            # NÚT XUẤT EXCEL
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Comparison')
            st.download_button(
                label="📥 XUẤT FILE EXCEL SO SÁNH",
                data=output.getvalue(),
                file_name=f"So_sanh_{st.session_state.sel_code}.xlsx",
                mime="application/vnd.ms-excel"
            )
