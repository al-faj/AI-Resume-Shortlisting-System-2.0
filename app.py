import sqlite3
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

app = Flask(__name__)
app.secret_key = "secret_key_for_post"

# =========================
# DATABASE
# =========================
def get_db_connection():
    conn = sqlite3.connect("resumes.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            score REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text.strip()

# =========================
# ML FUNCTION
# =========================
def calculate_similarity(resume_file, job_description):
    resume_text = extract_text_from_pdf(resume_file)

    if resume_text == "" or job_description.strip() == "":
        return 0.0

    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    score = None

    if request.method == "POST":
        job_description = request.form.get("job_description")
        resumes_files = request.files.getlist("resume")

        if job_description and resumes_files:
            for resume in resumes_files:
                if resume.filename != "":
                    score = calculate_similarity(resume, job_description)

                    conn = get_db_connection()
                    conn.execute(
                        "INSERT INTO resumes (filename, score) VALUES (?, ?)",
                        (resume.filename, score)
                    )
                    conn.commit()
                    conn.close()

    conn = get_db_connection()
    resumes = conn.execute(
        "SELECT * FROM resumes ORDER BY score DESC LIMIT 5"
    ).fetchall()
    conn.close()

    return render_template("index.html", resumes=resumes, score=score)

# =========================
# DELETE (405 FIX)
# =========================
@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):
    conn = get_db_connection()
    conn.execute("DELETE FROM resumes WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

