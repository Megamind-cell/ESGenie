import sqlite3
import pandas as pd
import json

# Connect to local DB file
conn = sqlite3.connect("chat_logs.db")  # Use full path if needed
c = conn.cursor()
c.execute("SELECT question, answer, sources, timestamp FROM chat_history ORDER BY timestamp DESC")
rows = c.fetchall()
conn.close()

# Process rows into a DataFrame
data = []
for q, a, s, t in rows:
    sources = ", ".join([
        f"{src['document']} (page {src['page']})"
        for src in json.loads(s)
    ])
    data.append({
        "Timestamp": t,
        "Question": q,
        "Answer": a,
        "Sources": sources
    })

df = pd.DataFrame(data)

# Export to Excel
df.to_excel("chat_history_log.xlsx", index=False)
print("Exported chat history to 'chat_history_log.xlsx'")
