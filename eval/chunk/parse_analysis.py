from src.client.mongodb_config import MongoConfig
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

col = MongoConfig.get_collection('manual_text')
docs = list(col.find())

def wc(text):
    return len(text.split())

counts = [wc(d['page_content']) for d in docs]
pages  = [d['metadata'].get('page') for d in docs if d.get('metadata')]

# --- Basic stats ---
print("═" * 60)
print(f"{'BASIC STATS':^60}")
print("═" * 60)
print(f"Total docs   : {len(docs)}")
print(f"Avg words    : {sum(counts)/len(counts):.0f}")
print(f"Median words : {sorted(counts)[len(counts)//2]}")
print(f"Max words    : {max(counts)}")
print(f"Min words    : {min(counts)}")
print(f"Pages        : {min(p for p in pages if p)} ~ {max(p for p in pages if p)}")

# --- Dump all chunks to file ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"chunk_dump_{timestamp}.txt"

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    filename
)

docs_sorted = sorted(docs, key=lambda d: (d['metadata'].get('page', 0)))

with open(output_path, "w", encoding="utf-8") as f:
    for i, doc in enumerate(docs_sorted):
        meta = doc.get('metadata', {})
        pc   = doc['page_content']
        f.write(f"{'='*60}\n")
        f.write(f"[{i+1}/{len(docs)}]  page={meta.get('page','?')}  words={wc(pc)}\n")
        f.write(f"{'─'*60}\n")
        f.write(pc + "\n")

print(f"\nDumped {len(docs)} chunks → {output_path}")

# --- Histogram ---
bins = [0, 20, 50, 100, 150, 200, 300, 400, 500, max(counts) + 1]
labels = [f"{bins[i]}–{bins[i+1]-1}" for i in range(len(bins)-1)]
hist, _ = np.histogram(counts, bins=bins)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(labels, hist, color="#4C9BE8", edgecolor="white", linewidth=0.6)

# value labels on top of each bar
for bar, val in zip(bars, hist):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=9)

# reference lines
avg = sum(counts) / len(counts)
med = sorted(counts)[len(counts)//2]
ax.axvline(x=labels.index(next(l for l in labels if avg < int(l.split('–')[1]))),
           color='red', linestyle='--', alpha=0.5, label=f'approx avg ({avg:.0f})')

ax.set_title(f"Chunk Word Count Distribution  (n={len(docs)})", fontsize=13, pad=12)
ax.set_xlabel("Word Count Range", fontsize=11)
ax.set_ylabel("Number of Chunks", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, max(hist) * 1.15)
plt.xticks(rotation=15)
plt.tight_layout()

chart_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"chunk_dist_{timestamp}.png"
)
plt.savefig(chart_path, dpi=150)
print(f"Chart saved → {chart_path}")
plt.show()