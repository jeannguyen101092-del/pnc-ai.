# ==========================================================
# AI FASHION PRO V15 - FULL PDF + EXCEL + QA SYSTEM
# ==========================================================

import streamlit as st
import fitz, re
import pandas as pd
from difflib import SequenceMatcher
from supabase import create_client

# ================== CONFIG ==================
SUPABASE_URL = "https://ewqqodsfvlvnrzsylawy.supabase.co"
SUPABASE_KEY = "sb_publishable_yxioECJT07sMQWL_rtSyFg_vJ1DF2ri"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================== UI ==================
st.set_page_config(layout="wide")
st.title("AI Fashion Pro V15 - QA System")

st.sidebar.header("📦 QUẢN LÝ KHO")
size = st.sidebar.selectbox("Chọn size (hoặc AUTO)", ["AUTO","0","2","4","6","8","10","12","14","16","XS","S","M","L","XL"])

col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
with col2:
    excel_file = st.file_uploader("Upload Excel định mức", type=["xlsx"])

# ================== PDF PARSE ==================
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


def extract_pdf_spec(pdf_file, target_size):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    specs = {}

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        header = detect_size_header(lines)
        size_index = find_size_index(header, target_size) if header else None

        for i, line in enumerate(lines):
            if re.match(r'^\d+\.\d+[A-Z]', line.strip()):
                try:
                    desc = lines[i+1].strip()
                    for j in range(i+2, i+8):
                        nums = re.findall(r'\d+\.?\d*', lines[j])
                        if nums:
                            idx = size_index if size_index is not None and size_index < len(nums) else len(nums)//2
                            val = float(nums[idx])
                            key = f"{line.strip()} {desc}".upper()
                            specs[key] = val
                            break
                except:
                    pass

    return specs

# ================== EXCEL PARSE ==================
def extract_excel_spec(excel_file):
    df = pd.read_excel(excel_file).fillna("")
    specs = {}

    # detect size column
    header = " ".join([str(c) for c in df.columns]).upper()
    sizes = re.findall(r'[A-Z]+|\d+', header)

    for _, row in df.iterrows():
        try:
            key = str(row[0]).strip().upper()
            nums = [x for x in row if isinstance(x, (int, float))]

            if key and nums:
                val = nums[len(nums)//2]  # AUTO size
                specs[key] = float(val)
        except:
            pass

    return specs

# ================== LOAD DB ==================
def load_db():
    return supabase.table("ai_data").select("*").execute().data

# ================== MATCH ==================
def find_best_match(specs, db):
    best, best_score = None, 0
    for item in db:
        db_spec = item.get("spec_json", {})
        score = sum(1 for k1 in specs for k2 in db_spec if SequenceMatcher(None, k1, k2).ratio() > 0.6)
        if score > best_score:
            best_score = score
            best = item
    return best

# ================== MAIN ==================
if pdf_file:
    pdf_specs = extract_pdf_spec(pdf_file, size)
    excel_specs = extract_excel_spec(excel_file) if excel_file else {}

    st.write("DEBUG PDF:", pdf_specs)
    st.write("DEBUG EXCEL:", excel_specs)

    if pdf_specs:
        db = load_db()
        target = find_best_match(pdf_specs, db)

        if target:
            st.success(f"Match: {target['file_name']}")
            db_spec = target.get("spec_json", {})

            rows = []
            for k, v in pdf_specs.items():
                db_val = None
                for dk in db_spec:
                    if SequenceMatcher(None, k, dk).ratio() > 0.6:
                        db_val = db_spec[dk]
                        break

                ex_val = excel_specs.get(k)

                diff_db = None if db_val is None else round(v - db_val, 2)
                diff_ex = None if ex_val is None else round(v - ex_val, 2)

                # QA FLAG
                status = "OK"
                if diff_db and abs(diff_db) > 0.5:
                    status = "NG_DB"
                if diff_ex and abs(diff_ex) > 0.5:
                    status = "NG_EXCEL"

                rows.append({
                    "POM": k,
                    "PDF": v,
                    "EXCEL": ex_val,
                    "DB": db_val,
                    "DIFF_PDF_DB": diff_db,
                    "DIFF_PDF_EXCEL": diff_ex,
                    "QA": status
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # EXPORT
            file = "QA_Report.xlsx"
            df.to_excel(file, index=False)
            with open(file, "rb") as f:
                st.download_button("📥 Xuất báo cáo QA", f, file_name=file)

        else:
            st.error("❌ Không tìm thấy mẫu giống")
    else:
        st.error("❌ Không đọc được PDF")
