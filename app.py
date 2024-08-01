from flask import Flask, request, jsonify, send_file, send_from_directory
import arxiv
from datetime import datetime, timedelta, timezone
import json
import os
from gptquery import GPT

app = Flask(__name__, static_url_path="")

# Configuration
MAX_RESULTS = 1000
OPENAI_KEY = "sk-proj-dRjv6kFVDwCtN70Fl8vZT3BlbkFJerxWWxCqZp5L6u2q7Lqd"  # Add your OpenAI key here
MODEL_NAME = "gpt-4-turbo"
PAPERS_PER_PAGE = 10

# Global variables
paper_data = []


def fetch_recent_papers():
    search = arxiv.Search(
        query="cat:cs.LG",
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    papers = []
    yesterday = datetime.now(timezone.utc) - timedelta(hours=24)

    for result in client.results(search):
        if result.published <= yesterday:
            break
        papers.append(
            {
                "title": result.title,
                "authors": ", ".join(str(author) for author in result.authors),
                "abstract": result.summary,
                "arxiv_id": result.entry_id,
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category,
            }
        )

    return papers


def evaluate_relevance(paper, preferences):
    llm = GPT(
        model_name=MODEL_NAME,
        task_prompt_text="""
        You are an AI assistant helping an AI researcher go through relevant papers.
        The researcher has the following preferences: {preferences}
        Now consider the following article:
        Title: {title}
        Abstract: {abstract}
        Is this paper relevant? Reply with RELEVANT or UNRELATED.
        """,
        oai_key=OPENAI_KEY,
        logging_path="verdicts.json",
        max_num_tokens=128,
    )

    result = llm([{**paper, "preferences": preferences}])[0]
    return "RELEVANT" in result["response"]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/fetch_papers", methods=["POST"])
def fetch_papers():
    global paper_data
    preferences = request.form["preferences"]
    papers = fetch_recent_papers()

    relevant_papers = []
    for paper in papers:
        if evaluate_relevance(paper, preferences):
            paper["votes"] = 0
            paper["prompt"] = preferences
            paper["model"] = MODEL_NAME
            relevant_papers.append(paper)

    paper_data = relevant_papers
    return jsonify(relevant_papers)


@app.route("/get_page", methods=["GET"])
def get_page():
    page = int(request.args.get("page", 1))
    start = (page - 1) * PAPERS_PER_PAGE
    end = start + PAPERS_PER_PAGE
    return jsonify(paper_data[start:end])


@app.route("/vote", methods=["POST"])
def vote():
    arxiv_id = request.form["arxiv_id"]
    vote_type = request.form["vote_type"]

    for paper in paper_data:
        if paper["arxiv_id"] == arxiv_id:
            paper["votes"] += 1 if vote_type == "up" else -1
            break

    return jsonify(success=True)


@app.route("/download_json")
def download_json():
    formatted_data = json.dumps(paper_data, indent=2)
    with open("paper_data.json", "w") as f:
        f.write(formatted_data)
    return send_file("paper_data.json", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
