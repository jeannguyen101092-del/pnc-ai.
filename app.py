# ==========================================================
# AI FASHION PRO V14 - AUTO SIZE + SMART PARSE (GIỮ NGUYÊN UI)
# ==========================================================

import streamlit as st
import fitz, re
import pandas as pd
from difflib import SequenceMatcher
from supabase import create_client

# ================== CONFIG ==================
URL= "https://ewqqodsfvlvnrzsylawy.supabase.co"
KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================== UI (GIỮ NGUYÊN) ==================
st.set_page_config(layout="wide")
st.title("AI Fashion Pro V14")

st.sidebar.header("📦 QUẢN LÝ KHO")
size = st.sidebar.selectbox("Chọn size (hoặc AUTO)", ["AUTO","0","2","4","6","8","10","12","14","16","XS","S","M","L","XL"])

col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
with col2:
    excel_file = st.file_uploader("Upload Excel", type=["xlsx"])

# ================== SMART SIZE DETECT ==================
def detect_size_header(lines):
    for line in lines:
        if re.search(r'(\d+\s+){3,}', line) or re.search(r'XS|S|M|L|XL', line):
            return line
    return None


def find_size_index(header, target_size):
    sizes = re.findall(r'[A-Z]+|\d+', header)
    if target_size == "AUTO":
        return len(sizes)//2
    for i, s in enumerate(sizes):
        if s == target_size:
            return i
    return None

# ================== PARSE PDF V14 ==================
def extract_spec(pdf_file, target_size):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    specs = {}

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        header = detect_size_header(lines)
        size_index = None

        if header:
            size_index = find_size_index(header, target_size)

        for i, line in enumerate(lines):
            if re.match(r'^\d+\.\d+[A-Z]', line.strip()):
                try:
                    desc = lines[i+1].strip()

                    for j in range(i+2, i+8):
                        nums = re.findall(r'\d+\.?\d*', lines[j])

                        if nums:
                            if size_index is None or size_index >= len(nums):
                                idx = len(nums)//2
                            else:
                                idx = size_index

                            val = float(nums[idx])
                            key = f"{line.strip()} {desc}"
                            specs[key] = val
                            break
                except:
                    pass

    return specs

# ================== LOAD DB ==================
def load_db():
    res = supabase.table("ai_data").select("*").execute()
    return res.data

# ================== MATCH ==================
def find_best_match(specs, db):
    best, best_score = None, 0
    for item in db:
        db_spec = item.get("spec_json", {})
        score = 0
        for k1 in specs:
            for k2 in db_spec:
                if SequenceMatcher(None, k1, k2).ratio() > 0.6:
                    score += 1
        if score > best_score:
            best_score = score
            best = item
    return best

# ================== MAIN ==================
if pdf_file:
    specs = extract_spec(pdf_file, size)

    st.write("DEBUG SPEC:", specs)

    if specs:
        db = load_db()
        target = find_best_match(specs, db)

        if target:
            st.success(f"Match: {target['file_name']}")

            rows = []
            db_spec = target.get("spec_json", {})

            for k, v in specs.items():
                db_val = None
                for dk in db_spec:
                    if SequenceMatcher(None, k, dk).ratio() > 0.6:
                        db_val = db_spec[dk]
                        break

                rows.append({
                    "POM": k,
                    "PDF": v,
                    "DB": db_val,
                    "DIFF": None if db_val is None else round(v - db_val, 2)
                })

            df = pd.DataFrame(rows)
            st.dataframe(df)

            # EXPORT EXCEL (GIỮ NGUYÊN)
            output = "compare.xlsx"
            df.to_excel(output, index=False)

            with open(output, "rb") as f:
                st.download_button("📥 Xuất Excel", f, file_name="compare.xlsx")

        else:
            st.error("❌ Không tìm thấy mẫu giống")
    else:
        st.error("❌ Không đọc được spec từ PDF")
