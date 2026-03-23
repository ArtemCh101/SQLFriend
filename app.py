import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import sqlite3
import pandas as pd
import re

st.set_page_config(page_title="SQLFriend: Text-to-SQL", layout="wide")

@st.cache_resource
def load_resource():
    model_id = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    lora_path = "sql_model_lora"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    with open("tables.json", "r") as f:
        tables = json.load(f)
    
    db_map = {}
    for db in tables:
        schema = []
        for i, tab_name in enumerate(db['table_names_original']):
            cols = [c[1] for c in db['column_names_original'] if c[0] == i]
            schema.append(f"{tab_name}({', '.join(cols)})")
        db_map[db['db_id']] = " | ".join(schema)
        
    return model, tokenizer, db_map

def clean_sql(text):
    clean = re.sub(r"```sql|```", "", text)
    select_match = re.search(r"\bSELECT\b", clean, re.IGNORECASE)
    if select_match:
        clean = clean[select_match.start():]

    return clean.split(';')[0].split('\n\n')[0].strip()

model, tokenizer, db_map = load_resource()

st.title("SQLFriend: Natural Language to SQL")


TRAIN_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Convert the following natural language question into a valid SQL query using the provided database schema.
Schema: {}

### Input:
{}

### Response:
"""

col1, col2 = st.columns([1, 3])

with col1:
    db_list = sorted(list(db_map.keys()))
    selected_db = st.selectbox("Database ID", db_list, index=db_list.index("concert_singer") if "concert_singer" in db_list else 0)
    st.info(f"Schema (Injected): \n\n {db_map[selected_db]}")

with col2:
    user_query = st.text_input("Ask a question about the data:", "Show all singer names and their countries")
    
    if st.button("Run Query"):
        prompt = TRAIN_PROMPT_TEMPLATE.format(db_map[selected_db], user_query)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        raw_gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        sql = clean_sql(raw_gen)
        
        st.subheader("Generated SQL")
        if sql:
            st.code(sql, language="sql")
            
            try:
                db_path = f"database/{selected_db}/{selected_db}.sqlite"
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(sql, conn)
                st.subheader("Results")
                st.dataframe(df)
                conn.close()
            except Exception as e:
                st.error(f"Execution Error: {e}")
        else:
            st.warning("The model generated an empty response. Check if the schema was properly loaded.")