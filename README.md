**README.md**

# Tamil Bible Chatbot

This is a Tamil Bible-based AI chatbot built using Google Gemini and FAISS. It reads a cleaned Tamil Bible JSONL file, chunks the content, creates semantic embeddings, and lets you ask questions in Tamil. The chatbot responds with relevant verses in a friendly, supportive manner.

## Features
- Tamil language support
- Embedding-based semantic search
- Answers grounded in Tamil Bible content
- Gemini 3.5 Flash model for natural responses

## Requirements
Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  Setup
1. Clone or download this repository
2. Place your `cleaned_tamil_bible_verses.jsonl` file in the project directory
3. Replace the `GEMINI_API_KEY` value in `main.py` with your actual Gemini API key

##  Run the Chatbot
```bash
python main.py
```

Then type your questions in Tamil!

```bash
🙋‍♀️= உங்கள் கேள்வி (exit என்றால் முடியும்): இன்று நான் சோகமாக இருக்கிறேன். தேவன் என்ன சொல்கிறார்?
```

##  File Format
The `cleaned_tamil_bible_verses.jsonl` must be a JSONL file with this format per line:
```json
{"book": "1 அரசர்கள்", "chapter": 2, "verse": 1, "text": "தாவீதுராஜா..."}
```




