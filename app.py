import streamlit as st
import os, fitz, io, pdfplumber, re, pandas as pd
import numpy as np
from PIL import Image
import torch, base64, tempfile
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# ==========================================
# 1. KẾT NỐI (TỰ ĐIỀN)
# ==========================================
URL =  "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase: Client = create_client(URL, KEY)

st.set_page_config(layout="wide", page_title="AI Fashion Pro V6", page_icon="👔")

# ==========================================
# LOAD AI MODEL
# ==========================================
@st.cache_resource
def load_ai():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model.eval()

ai_brain = load_ai()

# ==========================================
# CLASSIFY
# ==========================================
def advanced_classify(specs, text_content, file_name):
    txt = (text_content + " " + file_name).upper()
    inseam = specs.get('INSEAM', 0)

    # ===== ƯU TIÊN NHẬN DIỆN QUẦN =====

    # QUẦN YẾM
    if 'BIB' in txt or 'FRONT BIB WIDTH' in txt:
        return "QUẦN YẾM"

    # QUẦN CARGO
    if 'CARGO' in txt or 'FRONT CARGO POCKET' in txt:
        if 'JOGGER' in txt or 'ELASTIC' in txt:
            return "QUẦN CARGO LƯNG THUN"
        return "QUẦN TÚI CARGO"

    # QUẦN SHORT (ưu tiên trước)
    if ('SHORT' in txt or 'SKORT' in txt) or (inseam > 0 and inseam <= 11):
        return "QUẦN SHORT"

    # QUẦN DÀI
    if ('PANT' in txt or 'TROUSER' in txt) or (inseam >= 25):
        return "QUẦN DÀI"

    # ===== KHÁC =====

    if 'DRESS' in txt:
        return "ĐẦM"

    if 'SKIRT' in txt:
        return "VÁY"

    # ÁO (dựa vào thông số đặc trưng)
    if 'CHEST WIDTH' in txt or 'FRONT LENGTH' in txt or 'HPS' in txt:
        return "ÁO"

    return "KHÁC"

# ==========================================
# PARSE VALUE
# ==========================================
def parse_val(t):
    try:
        found = re.findall(r'(\d+\s\d+/\d+|\d+/\d+|\d+\.\d+|\d+)', str(t))
        if not found: return 0
        v = found[0]
        if ' ' in v:
            p = v.split()
            return float(p[0]) + eval(p[1])
        return eval(v) if '/' in v else float(v)
    except:
        return 0

# ==========================================
# ĐỌC PDF
# ==========================================
def get_data(pdf_path):
    try:
        specs, all_texts = {}, ""
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: all_texts += t + " "
                tables = p.extract_tables()
                for table in tables:
                    for r in table:
                        if r and len(r) >= 2:
                            val = parse_val(r[-1])
                            pom = " ".join([str(x) for x in r[:-1] if x]).strip().upper()
                            if val > 0 and len(pom) > 3:
                                specs[pom] = val

        doc = fitz.open(pdf_path)
        try:
            pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        except Exception as e:
            st.error(f"Lỗi render ảnh: {e}")
            return None

        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        cat = advanced_classify(specs, all_texts, os.path.basename(pdf_path))

        return {
            "spec": specs,
            "img_b64": img_b64,
            "img_bytes": img_bytes,
            "cat": cat
        }
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
        return None

# ==========================================
# SIDEBAR - KHO
# ==========================================
with st.sidebar:
    st.header("📦 QUẢN TRỊ KHO")

    try:
        res_db = supabase.table("ai_data").select("file_name", count="exact").execute()
        st.metric("Tổng mẫu", res_db.count if res_db.count else 0)
    except Exception as e:
        st.error(f"Lỗi DB: {e}")

    up_bulk = st.file_uploader("Nạp PDF", accept_multiple_files=True)

    if up_bulk and st.button("🚀 NẠP KHO"):
        skipped = 0
        success = 0

        for f in up_bulk:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                path = tmp.name

            d = get_data(path)

            # ===== KIỂM TRA ĐIỀU KIỆN =====
            if not d or not d.get('img_bytes') or not d.get('spec') or len(d['spec']) == 0:
                skipped += 1
                os.remove(path)
                continue

            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            try:
                with torch.no_grad():
                    vec = ai_brain(tf(Image.open(io.BytesIO(d['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy().tolist()

                # kiểm tra vector
                if not vec or len(vec) == 0:
                    skipped += 1
                    os.remove(path)
                    continue

                # ===== INSERT DB =====
                supabase.table("ai_data").upsert({
                    "file_name": f.name,
                    "vector": vec,
                    "spec_json": d['spec'],
                    "img_base64": d['img_b64'],
                    "category": d['cat']
                }).execute()

                success += 1

            except Exception as e:
                st.error(f"Lỗi xử lý {f.name}: {e}")
                skipped += 1

            # ===== XOÁ FILE SAU KHI XỬ LÝ =====
            try:
                os.remove(path)
            except:
                pass

        st.success(f"✅ Nạp thành công: {success} file")
        st.warning(f"⚠️ Bỏ qua: {skipped} file (thiếu ảnh / thiếu spec)")
        st.rerun()

# ==========================================
# SO SÁNH V7 PRO
# ==========================================
st.title("👔 AI Fashion Pro V7 PRO")

up_test = st.file_uploader("Upload PDF", type="pdf")

if up_test:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(up_test.getbuffer())
        path = tmp.name

    target = get_data(path)

    if target:
        st.success(f"Nhận diện: {target['cat']}")

        try:
            db = supabase.table("ai_data").select("*").eq("category", target['cat']).execute()
        except Exception as e:
            st.error(f"Lỗi DB: {e}")
            st.stop()

        if not db.data:
            st.warning("Kho chưa có mẫu cùng loại")
        else:
            tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            with torch.no_grad():
                v_test = ai_brain(tf(Image.open(io.BytesIO(target['img_bytes'])).convert('RGB')).unsqueeze(0)).flatten().numpy()

            sims = []

            for i in db.data:
                if i.get('vector'):
                    try:
                        vec_db = np.array(i['vector']).reshape(1, -1)
                        vec_test = v_test.reshape(1, -1)
                        sim_img = float(cosine_similarity(vec_test, vec_db)[0][0])

                        # ===== SO SÁNH SPEC =====
                        spec_score = 0
                        count = 0
                        for k in target['spec']:
                            if k in i['spec_json']:
                                v1 = target['spec'][k]
                                v2 = i['spec_json'][k]
                                if v1 > 0 and v2 > 0:
                                    diff = abs(v1 - v2) / max(v1, v2)
                                    spec_score += (1 - diff)
                                    count += 1

                        sim_spec = (spec_score / count) if count > 0 else 0

                        # ===== COMBINE SCORE =====
                        final_score = (sim_img * 0.6 + sim_spec * 0.4) * 100

                        sims.append({
                            "name": i['file_name'],
                            "sim": final_score,
                            "img_sim": sim_img * 100,
                            "spec_sim": sim_spec * 100,
                            "spec": i['spec_json'],
                            "img": i['img_base64']
                        })
                    except:
                        continue

            if sims:
                sims = sorted(sims, key=lambda x: x['sim'], reverse=True)

                # ===== HIỂN THỊ TOP 5 =====
                st.subheader("🏆 TOP MẪU TƯƠNG ĐỒNG")

                top_n = sims[:5]

                for idx, item in enumerate(top_n):
                    with st.expander(f"#{idx+1} - {item['name']} | Tổng: {item['sim']:.1f}% | Ảnh: {item['img_sim']:.1f}% | Spec: {item['spec_sim']:.1f}%"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(target['img_bytes'], caption="Mẫu mới", use_container_width=True)
                        with c2:
                            st.image(base64.b64decode(item['img']), caption=item['name'], use_container_width=True)

                        # bảng so sánh
                        diff_list = []
                        poms = sorted(list(set(target['spec'].keys()) | set(item['spec'].keys())))

                        for p in poms:
                            v1 = target['spec'].get(p, 0)
                            v2 = item['spec'].get(p, 0)
                            diff_list.append({
                                "POM": p,
                                "Mẫu mới": v1,
                                "Mẫu kho": v2,
                                "Diff": round(v1 - v2, 2)
                            })

                        df = pd.DataFrame(diff_list)
                        st.dataframe(df)

                # ===== EXPORT EXCEL TOP =====
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                    for i, item in enumerate(top_n):
                        df = pd.DataFrame(diff_list)
                        df.to_excel(writer, sheet_name=f"Top_{i+1}", index=False)

                st.download_button("📥 Tải Excel TOP", out.getvalue(), "Top_compare.xlsx")

# DEBUG
# st.write(target)
# st.write(target)
