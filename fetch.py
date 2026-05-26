import requests
from bs4 import BeautifulSoup
import os
import time
from tqdm import tqdm

# 66 Books of the Bible (Tamil names in correct order)
BOOKS = [
    "ஆதியாகமம்", "யாத்திராகமம்", "லேவியராகமம்", "எண்ணாகமம்", "உபாகமம்",
    "யோசுவா", "நியாயாதிபதிகள்", "ரூத்", "1 சாமுவேல்", "2 சாமுவேல்",
    "1 அரசர்கள்", "2 அரசர்கள்", "1 நாளாகமம்", "2 நாளாகமம்",
    "எஸ்றா", "நெகேமியா", "எஸ்தர்", "யோபு", "சங்கீதம்", "நீதிமொழிகள்",
    "பிரசங்கி", "உன்னதப்பாட்டு", "எசாயா", "எரேமியா", "விலாப்பாட்டு",
    "எசேக்கியேல்", "தானியேல்", "ஓசியா", "யோவேல்", "ஆமோஸ்", "ஒபதியா",
    "யோனா", "மீக்கா", "நாகூம்", "ஆபக்கூக்", "செப்பனியா", "ஆகாய்",
    "சகரியா", "மல்கி", "மத்தேயு", "மாற்கு", "லூக்கா", "யோவான்",
    "அப்போஸ்தலர்", "ரோமர்", "1 கொரிந்தியர்", "2 கொரிந்தியர்", "கலாத்தியர்",
    "எபேசியர்", "பிலிப்பியர்", "கொலோசெயர்", "1 தெசலோனிக்கேயர்",
    "2 தெசலோனிக்கேயர்", "1 தீமோத்தேயு", "2 தீமோத்தேயு", "தீத்து", "பிலேமோன்",
    "எபிரெயர்", "யாக்கோபு", "1 பேதுரு", "2 பேதுரு", "1 யோவான்",
    "2 யோவான்", "3 யோவான்", "யூதா", "வெளிப்படுத்தல்"
]

BASE_URL = "https://tamilbible.org/tamil"

# Create folder to store books
os.makedirs("tamil_bible_books", exist_ok=True)

# Loop over all 66 books
for i, book_name in enumerate(tqdm(BOOKS, desc="Scraping Bible")):
    book_number = f"{i+1:02}"  # e.g., 01, 02, ..., 66
    chapter = 1
    full_text = ""

    while True:
        url = f"{BASE_URL}/{book_number}/{chapter}.htm"
        response = requests.get(url)
        response.encoding = "utf-8"  # ✅ Fix encoding issue

        # Stop when chapter doesn't exist
        if "It looks like nothing was found" in response.text or response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        verses = soup.select("p")

        if not verses:
            break

        # Add chapter and verses to the book text
        full_text += f"\n\nஅதிகாரம் {chapter}\n"
        full_text += "\n".join(verse.get_text(strip=True) for verse in verses)

        chapter += 1
        time.sleep(0.2)  # Be gentle to the server

    # Save the book to a file
    with open(f"tamil_bible_books/{book_name}.txt", "w", encoding="utf-8") as f:
        f.write(full_text.strip())

    print(f"✅ Saved: {book_name}")
