import streamlit as st
import io, fitz, pdfplumber, re, pandas as pd, numpy as np
import torch, hashlib, time, uuid, json
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from difflib import get_close_matches

# ================= 1. CONFIGURATION =================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"

supabase = create_client(URL, KEY)
BUCKET = "fashion-imgs"

st.set_page_config(layout="wide", page_title="PPJ AI DNA Auditor Pro", page_icon="👔")

if 'sel_audit' not in st.session_state: st.session_state['sel_audit'] = None
if 'ver_results' not in st.session_state: st.session_state['ver_results'] = None
if 'up_key' not in st.session_state: st.session_state['up_key'] = 0

# ================= 2. DNA LOGIC (CORE SO SÁNH) =================

def get_dna_logic(item):
    """Suy luận DNA từ dữ liệu có sẵn (Dùng cho cả file cũ và mới)"""
    if item.get("dna_json"): return item["dna_json"]
    
    # Suy luận từ 1000 file cũ dựa trên tên điểm đo (spec_json)
    specs = item.get("spec_json", {})
    all_p = ""
    for sz in specs: all_p += " ".join(specs[sz].keys()).upper()
    
    return {
        "pockets": {
            "front": "slanted" if "SLANT" in all_p else ("curved" if "CURVE" in all_p else "patch"),
            "back": "patch" if "BACK" in all_p else "none",
            "extra": "coin" if "COIN" in all_p else ("cargo" if "CARGO" in all_p else "none")
        },
        "construction": {"closure": "zipper" if "ZIP" in all_p else "button", "belt_loop": "yes"},
        "fit": {"shape": "regular"}, "material": {"fabric": "woven"}, "appearance": {"color": "unknown"}
    }

def calculate_dna_similarity(dna1, dna2):
    """Tính điểm theo trọng số: Túi 40%, Kết cấu 20%, Form 15%, Chất liệu 15%, Ngoại quan 10%"""
    if not dna1 or not dna2: return 0.0
    score = 0.0
    # 1. Túi (40%)
    p_score = 0
    if dna1["pockets"]["front"] == dna2["pockets"]["front"]: p_score += 0.4
    if dna1["pockets"]["back"] == dna2["pockets"]["back"]: p_score += 0.4
    if dna1["pockets"]["extra"] == dna2["pockets"]["extra"]: p_score += 0.2
    score += (p_score * 0.4)
    # 2. Kết cấu (20%) + 3. Form (15%) + 4. Chất liệu (15%) + 5. Ngoại quan (10%)
    if dna1["construction"] == dna2["construction"]: score += 0.2
    if dna1["fit"] == dna2["fit"]: score += 0.15
    if dna1["material"] == dna2["material"]: score += 0.15
    if dna1["appearance"] == dna2["appearance"]: score += 0.10
    return score

# ================= 3. AI & SCRAPER (GIỮ NGUYÊN PHẦN CHUẨN CỦA BẠN) =================

def parse_val(t):
    try:
        t = str(t).replace('"', '').strip().lower().replace(',', '.')
        if not t or any(x in t for x in ["wash", "color", "label", "page", "tol", "+", "-"]): return 0
        if re.match(r'^[a-z]\d+', t): return 0 
        mixed = re.match(r'(\d+)\s+(\d+)/(\d+)', t)
        if mixed: return float(mixed.group(1)) + int(mixed.group(2))/int(mixed.group(3))
        frac = re.match(r'(\d+)/(\d+)', t)
        if frac: return int(frac.group(1))/int(frac.group(2))
        num = re.findall(r"[-+]?\d*\.\d+|\d+", t)
        return float(num[0]) if num else 0
    except: return 0

def extract_full_data(file_content):
    if not file_content: return None
    all_specs, all_text = {}, ""
    SIZE_PATTERN = r'^(xs|s|m|l|xl|xxl|\d+|[a-z]?\d+-\d+|[a-z]?\d+\.\d+)$'
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = io.BytesIO(); Image.open(io.BytesIO(pix.tobytes("png"))).save(img_bytes, format="WEBP", quality=70)
        img_data = img_bytes.getvalue(); doc.close()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                all_text += (page.extract_text() or "") + " "
                words = page.extract_words()
                if not words: continue
                df_w = pd.DataFrame(words)
                df_w['y_grid'] = df_w['top'].round(0)
                size_cols = []
                for y, group in df_w.groupby('y_grid'):
                    line_txt = " ".join(group.sort_values('x0')['text']).lower()
                    if "size" in line_txt or "adopted" in line_txt:
                        for _, row in group.iterrows():
                            txt = row['text'].strip().lower()
                            if re.match(SIZE_PATTERN, txt) and txt not in ["tol", "um", "(+)", "(-)"]:
                                size_cols.append({"sz": txt.upper(), "x0": row['x0']-5, "x1": row['x1']+5})
                        if size_cols: break
                for y, group in df_w.groupby('y_grid'):
                    sorted_row = group.sort_values('x0')
                    line_txt = " ".join(sorted_row['text']).upper()
                    if any(x in line_txt for x in ["COVER", "IMAGE", "CONSTRUCTION"]): continue
                    pom_name = re.sub(r'[\d./\s]+$', '', " ".join(sorted_row[sorted_row['x1'] < 350]['text'])).strip()
                    if len(pom_name) > 3:
                        for col in size_cols:
                            cell = sorted_row[(sorted_row['x0'] >= col['x0']) & (sorted_row['x1'] <= col['x1'])]
                            if not cell.empty:
                                val = parse_val(" ".join(cell['text']))
                                if val > 0:
                                    if col['sz'] not in all_specs: all_specs[col['sz']] = {}
                                    all_specs[col['sz']][pom_name] = val
        
        # Tạo DNA từ văn bản PDF
        text = all_text.upper()
        dna = {
            "pockets": {
                "front": "slanted" if "SLANT" in text else ("curved" if "CURVE" in text or "JEAN POCKET" in text else "patch"),
                "back": "patch" if "PATCH" in text and "BACK" in text else ("welt" if "WELT" in text and "BACK" in text else "none"),
                "extra": "coin" if "COIN" in text else ("cargo" if "CARGO" in text else "none")
            },
            "construction": {"closure": "button" if "BUTTON FLY" in text else "zipper", "belt_loop": "yes"},
            "fit": {"shape": "slim" if "SLIM" in text else "regular"},
            "material": {"fabric": "denim" if "DENIM" in text else "woven"}, "appearance": {"color": "unknown"}
        }
        return {"all_specs": all_specs, "img": img_data, "dna": dna}
    except: return None

def to_excel(df_list, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, name in zip(df_list, sheet_names): df.to_excel(writer, index=False, sheet_name=str(name)[:31])
    return output.getvalue()

# ================= 4. SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: #1E3A8A; font-weight: bold;'>PPJ GROUP</h1>", unsafe_allow_html=True)
    res_count = supabase.table("ai_data").select("id", count="exact").execute()
    count = res_count.count or 0
    st.metric("Models in Repo", f"{count} SKUs")
    storage_mb = count * 0.08
    st.write(f"💾 **Storage:** {storage_mb:.1f}MB / 1024MB")
    st.progress(min(storage_mb/1024, 1.0))
    st.divider()
    new_files = st.file_uploader("Upload Tech-Packs", accept_multiple_files=True, key=f"sy_{st.session_state['up_key']}")
    if new_files and st.button("🚀 SYNCHRONIZE", use_container_width=True):
        prog = st.progress(0); p_text = st.empty()
        for i, f in enumerate(new_files):
            data = extract_full_data(f.getvalue())
            if data:
                f_hash = hashlib.md5(f.name.encode()).hexdigest()
                path = f"lib_{f_hash}.webp"
                supabase.storage.from_(BUCKET).upload(path, data['img'], {"content-type": "image/webp", "upsert": "true"})
                supabase.table("ai_data").upsert({
                    "id": str(uuid.UUID(f_hash)), "file_name": f.name, "spec_json": data['all_specs'], 
                    "dna_json": data['dna'], "image_url": supabase.storage.from_(BUCKET).get_public_url(path)
                }).execute()
            prog.progress((i+1)/len(new_files)); p_text.write(f"Processing: {i+1}/{len(new_files)}")
        st.session_state['up_key'] += 1; st.rerun()

# ================= 5. MAIN UI =================
st.title("👔 AI SMART AUDITOR (DNA LOGIC)")
mode = st.radio("Chế độ:", ["🔍 Audit Mode", "🔄 Version Control"], horizontal=True)

if mode == "🔍 Audit Mode":
    f_audit = st.file_uploader("Upload Target PDF:", type="pdf")
    if f_audit:
        target = extract_full_data(f_audit.getvalue())
        if target:
            res = supabase.table("ai_data").select("*").execute()
            if res.data:
                valid_matches = []
                for r in res.data:
                    repo_dna = get_dna_logic(r) # Tự suy luận DNA nếu là file cũ
                    sim = calculate_dna_similarity(target['dna'], repo_dna)
                    valid_matches.append({**r, "sim_score": sim})
                df_db = pd.DataFrame(valid_matches).sort_values('sim_score', ascending=False).head(3)
                cols = st.columns(4)
                cols[0].image(target['img'], caption="TARGET PDF", use_container_width=True)
                for i, (idx, row) in enumerate(df_db.iterrows()):
                    with cols[i+1]:
                        st.image(row['image_url'], caption=f"DNA Match: {row['sim_score']:.1%}")
                        if st.button(f"SELECT {i+1}", key=f"s_{idx}"): st.session_state['sel_audit'] = row
            if st.session_state['sel_audit']:
                sel = st.session_state['sel_audit']
                st.divider(); st.success(f"📈 So sánh với: **{sel['file_name']}**")
                audit_dfs, sheet_names = [], []
                for sz, t_specs in target['all_specs'].items():
                    with st.expander(f"SIZE: {sz}", expanded=True):
                        r_specs = sel['spec_json'].get(get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4)[0] if get_close_matches(sz, list(sel['spec_json'].keys()), 1, 0.4) else "", {})
                        rows = [{"Point": p, "Target": v, "Ref": r_specs.get(get_close_matches(p, list(r_specs.keys()), 1, 0.6)[0] if get_close_matches(p, list(r_specs.keys()), 1, 0.6) else "", 0)} for p, v in t_specs.items()]
                        for r in rows: r['Diff'] = f"{r['Target'] - r['Ref']:+.3f}"
                        df_sz = pd.DataFrame(rows); st.table(df_sz); audit_dfs.append(df_sz); sheet_names.append(sz)
                st.download_button("📥 Xuất Excel Audit", to_excel(audit_dfs, sheet_names), f"Audit_{sel['file_name']}.xlsx")

elif mode == "🔄 Version Control":
    st.subheader("🔄 So sánh 2 file PDF mới")
    c1, c2 = st.columns(2)
    f1, f2 = c1.file_uploader("Bản cũ (A):", type="pdf", key="v1"), c2.file_uploader("Bản mới (B):", type="pdf", key="v2")
    if f1 and f2:
        if st.button("⚡ Bắt đầu so sánh", use_container_width=True):
            d1, d2 = extract_full_data(f1.getvalue()), extract_full_data(f2.getvalue())
            if d1 and d2: st.session_state['ver_results'] = {"d1": d1, "d2": d2, "f1_name": f1.name, "f2_name": f2.name}
    if st.session_state.get('ver_results'):
        vr = st.session_state['ver_results']
        st.divider(); col_a, col_b = st.columns(2)
        col_a.image(vr['d1']['img'], caption=f"Bản A", use_container_width=True); col_b.image(vr['d2']['img'], caption=f"Bản B", use_container_width=True)
        all_sz = sorted(list(set(vr['d1']['all_specs'].keys()) | set(vr['d2']['all_specs'].keys())), key=lambda x: str(x))
        version_dfs, ver_sheets = [], []
        for sz in all_sz:
            with st.expander(f"SIZE: {sz}", expanded=True):
                s1, s2 = vr['d1']['all_specs'].get(sz, {}), vr['d2']['all_specs'].get(sz, {})
                rows = [{"Point": p, "Ver A": s1.get(p,0), "Ver B": s2.get(p,0), "Diff": f"{s2.get(p,0)-s1.get(p,0):+.3f}", "Status": "✅" if s1.get(p,0)==s2.get(p,0) else "⚠️"} for p in sorted(list(set(s1.keys()) | set(s2.keys())))]
                df_sz = pd.DataFrame(rows); st.table(df_sz); version_dfs.append(df_sz); ver_sheets.append(sz)
        st.download_button("📥 Xuất Excel So Sánh", to_excel(version_dfs, ver_sheets), "Comparison.xlsx")
