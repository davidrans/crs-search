import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer
import os

import re
from bs4 import BeautifulSoup

def parse_crs_html(filepath):
    """
    Parses Colorado Revised Statutes HTML structure.
    Returns a list of tuples: (Title_Name, Article_Name, Section_ID, Section_Content)
    """
    
    # 1. Open with the correct encoding (CRS HTML often uses windows-1252)
    with open(filepath, "r", encoding="cp1252", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    # 2. Extract the Main Title (e.g., "Title 18 Criminal Code")
    # Usually in the first H1 or Title tag
    main_title = "Unknown Title"
    h1 = soup.find("h1")
    if h1:
        main_title = h1.get_text(strip=True).replace("\n", " ")

    sections_data = []
    
    # State variables
    current_article = "General Provisions" # Default if no article found
    current_section_id = None
    current_buffer = []

    # Regex to find Section IDs (e.g., "18-1-101." or "18-1-102.5.")
    # Looks for: Start of line, digits-digits-digits, optional decimal, then a dot.
    section_pattern = re.compile(r"^\s*(\d+-\d+-\d+(?:\.\d+)?)\.?")

    # 3. Iterate through all paragraphs
    # The CRS HTML uses <p> tags for everything (Headers, Articles, Sections)
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True) # Replace &nbsp; with space
        
        # A. Detect Article Headers
        # They usually look like "ARTICLE 1" or "ARTICLE 3"
        if text.upper().startswith("ARTICLE") and len(text) < 50:
            current_article = text
            continue # Don't add the header to the previous section content

        # B. Detect Start of a New Section
        match = section_pattern.match(text)
        if match:
            # If we were already building a section, save it now
            if current_section_id:
                full_content = "\n".join(current_buffer).strip()
                if full_content:
                    sections_data.append((main_title, current_article, current_section_id, full_content))
            
            # Reset for the new section
            current_section_id = match.group(1) # e.g., "18-1-101"
            current_buffer = [text] # Start buffer with this paragraph
            
        # C. Continuation of Current Section
        elif current_section_id:
            # Optional: Filter out metadata to keep vector search clean
            if text.startswith("Source:") or text.startswith("Cross references:") or text.startswith("Law reviews:"):
                continue 
                
            current_buffer.append(text)

    # 4. Save the very last section (loop finishes before adding it)
    if current_section_id and current_buffer:
        full_content = "\n".join(current_buffer).strip()
        sections_data.append((main_title, current_article, current_section_id, full_content))

    return sections_data

DB_PATH = "data/docs.db"

def run_indexing():
    os.makedirs("data", exist_ok=True)
    
    # 1. Setup DB
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Create table with specific columns for Title, Article, Section
    db.execute("DROP TABLE IF EXISTS sections")
    db.execute("DROP TABLE IF EXISTS vec_sections")
    
    db.execute("""
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY,
            main_title TEXT,
            article TEXT,
            section_id TEXT,
            content TEXT
        )
    """)
    # 384 dimensions is standard for 'all-MiniLM-L6-v2'
    db.execute("CREATE VIRTUAL TABLE vec_sections USING vec0(embedding float[384])")

    print("Loading AI Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Parsing HTML...")
    # CALL THE PARSER
    all_sections = parse_crs_html(r"C:\Users\drans\Desktop\Colorado Revised Statutes 2024 Title 18 Criminal Code.html")
    print(f"Found {len(all_sections)} sections.")

    print("Vectorizing...")
    for main_title, article, sec_id, content in all_sections:
        
        # Combine fields for the embedding context (helps the AI understand hierarchy)
        # e.g. "Article 1. 18-1-101. The purpose of this code..."
        text_to_embed = f"{article}. {sec_id}. {content}"
        embedding = model.encode(text_to_embed)
        
        # Insert Text
        cur = db.execute(
            "INSERT INTO sections(main_title, article, section_id, content) VALUES(?, ?, ?, ?) RETURNING id", 
            (main_title, article, sec_id, content)
        )
        row_id = cur.fetchone()[0]
        
        # Insert Vector
        db.execute("INSERT INTO vec_sections(rowid, embedding) VALUES(?, ?)", (row_id, embedding))

    db.commit()
    print("Done!")

if __name__ == "__main__":
    # You might need to paste the parse_crs_html function here or import it
    run_indexing()