from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    send_from_directory,
    Response,
    stream_with_context,
)
import arxiv
from datetime import datetime, timedelta, timezone
import json
import os
from gptquery import GPT
import time

app = Flask(__name__, static_url_path="")

# Configuration
MAX_RESULTS = None
OPENAI_KEY = None
MODEL_NAME = "gpt-4-turbo"
PAPERS_PER_PAGE = 10

# Global variables
paper_data = []


def fetch_recent_papers(query="cat:cs.LG"):
    search = arxiv.Search(
        query=query, max_results=MAX_RESULTS, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    papers = []
    last_week = datetime.now(timezone.utc) - timedelta(days=7)

    for result in client.results(search):
        if result.published >= last_week:
            papers.append(
                {
                    "title": result.title,
                    "authors": ", ".join(str(author) for author in result.authors),
                    "abstract": result.summary,
                    "arxiv_id": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "primary_category": result.primary_category,
                    "votes": 0,
                }
            )
        else:
            break

    return papers


def evaluate_relevance(paper, preferences):
    try:
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
            logging_path="verdicts.jsonl",
            max_num_tokens=128,
        )

        result = llm([{**paper, "preferences": preferences}])[0]
        return "RELEVANT" in result["response"]
    except Exception as e:
        print(f"Error in evaluate_relevance: {str(e)}")
        return False


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/fetch_papers", methods=["POST"])
def fetch_papers():
    preferences = request.form["preferences"]
    is_annotation_mode = request.form.get("is_annotation_mode") == "true"

    def generate():
        papers = fetch_recent_papers()
        total_papers = len(papers)

        for i, paper in enumerate(papers):
            if is_annotation_mode or evaluate_relevance(paper, preferences):
                paper["votes"] = 0
                paper["prompt"] = preferences
                paper["model"] = MODEL_NAME
                paper_data.append(paper)

            progress = (i + 1) / total_papers * 100
            yield json.dumps(
                {"progress": progress, "current": i + 1, "total": total_papers}
            ) + "\n"

            # Add a small delay to avoid overwhelming the client
            time.sleep(0.1)

        yield json.dumps(
            {
                "done": True,
                "papers": paper_data[:PAPERS_PER_PAGE],
                "total": len(paper_data),
            }
        ) + "\n"

    return Response(stream_with_context(generate()), content_type="application/json")


@app.route("/get_page", methods=["GET"])
def get_page():
    page = int(request.args.get("page", 1))
    start = (page - 1) * PAPERS_PER_PAGE
    end = start + PAPERS_PER_PAGE
    return jsonify({"papers": paper_data[start:end], "total": len(paper_data)})


@app.route("/vote", methods=["POST"])
def vote():
    arxiv_id = request.form["arxiv_id"]
    vote_type = request.form["vote_type"]

    for paper in paper_data:
        if paper["arxiv_id"] == arxiv_id:
            if paper["votes"] == 1 and vote_type == "up":
                paper["votes"] = 1
            elif paper["votes"] == -1 and vote_type == "down":
                paper["votes"] = -1
            else:
                paper["votes"] += 1 if vote_type == "up" else -1
            break

    return jsonify({"success": True, "votes": paper["votes"]})


@app.route("/download_json")
def download_json():
    if not os.path.exists("paper_data.json"):
        with open("paper_data.json", "w") as f:
            json.dump([], f)

    with open("paper_data.json", "r+") as f:
        existing_data = json.load(f)
        existing_data.extend(paper_data)
        f.seek(0)
        json.dump(existing_data, f, indent=2)
        f.truncate()

    return send_file("paper_data.json", as_attachment=True)


@app.route("/update_settings", methods=["POST"])
def update_settings():
    global OPENAI_KEY, MAX_RESULTS

    new_openai_key = request.form.get("openaiKey")
    new_max_results = request.form.get("maxResults")

    if new_openai_key:
        OPENAI_KEY = new_openai_key
        os.environ["OPENAI_API_KEY"] = new_openai_key

    if new_max_results:
        try:
            MAX_RESULTS = int(new_max_results)
        except ValueError:
            return jsonify({"success": False, "message": "Invalid MAX_RESULTS value"})

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
