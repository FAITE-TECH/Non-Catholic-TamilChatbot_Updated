import json
import re

input_path = "tamil_bible_chunks.jsonl"
output_path = "cleaned_tamil_bible_verses.jsonl"

def split_verses(text):
    # This pattern splits just *before* a digit followed by Tamil letter
    parts = re.split(r'(?=\d{1,3}[அ-ஹ])', text.strip())
    result = []
    for part in parts:
        match = re.match(r'(\d{1,3})([அ-ஹ].*)', part)
        if match:
            verse_number = int(match.group(1))
            verse_text = match.group(2).strip()
            result.append((verse_number, verse_text))
    return result

cleaned = []

with open(input_path, "r", encoding="utf-8") as infile:
    lines = [json.loads(line) for line in infile]

for entry in lines:
    book = entry["book"]
    id_str = entry["id"]
    text = entry["text"]
    
    if "அதிகாரம்" in text:
        continue  # skip pure chapter headings

    # Extract chapter number from id like "ஆதியாகமம்_1"
    try:
        chapter = int(id_str.split("_")[1])
    except:
        continue

    # Skip junk
    if "Tamil Christian Assembly" in text:
        continue

    # Now split verses
    verses = split_verses(text)

    for verse_number, verse_text in verses:
        cleaned.append({
            "book": book,
            "chapter": chapter,
            "verse": verse_number,
            "text": verse_text
        })

# Write cleaned output
with open(output_path, "w", encoding="utf-8") as out:
    for verse in cleaned:
        json.dump(verse, out, ensure_ascii=False)
        out.write("\n")
