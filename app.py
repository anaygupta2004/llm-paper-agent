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
MAX_RESULTS = 100
OPENAI_KEY = None
MODEL_NAME = "gpt-4-turbo"
PAPERS_PER_PAGE = 10
DATE_RANGE = 7  # Default to 7 days
ARXIV_CATEGORIES = ["cs.LG"]  # Default to cs.LG (Machine Learning)

# Global variables
paper_data = []


def get_project_root():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up until we find the .git folder or reach the root
    while current_dir != "/":
        if ".git" in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # If we didn't find a .git folder, return the current script's directory
    return os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = get_project_root()
VERDICTS_FILE = os.path.join(PROJECT_ROOT, "verdicts.jsonl")


def fetch_recent_papers():
    search = arxiv.Search(
        query=" OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES),
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    papers = []
    date_limit = datetime.now(timezone.utc) - timedelta(days=DATE_RANGE)

    for result in client.results(search):
        if result.published >= date_limit:
            papers.append(
                {
                    "title": result.title,
                    "authors": ", ".join(str(author) for author in result.authors),
                    "abstract": result.summary,
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "pdf_url": result.pdf_url,
                    "abstract_url": result.entry_id,
                    "primary_category": result.primary_category,
                    "published_date": result.published.strftime("%Y-%m-%d"),
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
            Is this paper strongly relevant? If so reply with RELEVANT, if not reply NOT_ENOUGH_RELATED.
            """,
            oai_key=OPENAI_KEY,
            logging_path=VERDICTS_FILE,
            max_num_tokens=128,
        )

        result = llm([{**paper, "preferences": preferences}])[0]

        # Append the verdict to the file
        with open(VERDICTS_FILE, "a") as f:
            json.dump(
                {
                    "title": paper["title"],
                    "response": result["response"],
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )
            f.write("\n")

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
        global paper_data
        paper_data = []  # Reset paper_data
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
            if vote_type == "up":
                paper["votes"] += 1
            else:
                paper["votes"] -= 1
            break

    return jsonify({"success": True, "votes": paper["votes"]})


@app.route("/download_json")
def download_json():
    if not os.path.exists("data"):
        os.makedirs("data")

    is_annotation_mode = request.args.get("is_annotation_mode") == "true"
    filename = f"paper_data_{'annotation' if is_annotation_mode else 'relevance'}.json"
    filepath = os.path.join(PROJECT_ROOT, "data", filename)
    print(filepath)
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump([], f)

    with open(filepath, "r+") as f:
        existing_data = json.load(f)
        existing_data.extend(paper_data)
        f.seek(0)
        json.dump(existing_data, f, indent=2)
        f.truncate()

    return send_file(filepath, as_attachment=True)


@app.route("/update_settings", methods=["POST"])
def update_settings():
    global OPENAI_KEY, MAX_RESULTS, PAPERS_PER_PAGE, DATE_RANGE, ARXIV_CATEGORIES

    new_openai_key = request.form.get("openaiKey")
    new_max_results = request.form.get("maxResults")
    new_papers_per_page = request.form.get("papersPerPage")
    new_date_range = request.form.get("dateRange")
    new_arxiv_categories = request.form.get("arxivCategories")

    if new_openai_key:
        OPENAI_KEY = new_openai_key
        os.environ["OPENAI_API_KEY"] = new_openai_key

    if new_max_results:
        try:
            MAX_RESULTS = int(new_max_results)
        except ValueError:
            return jsonify({"success": False, "message": "Invalid MAX_RESULTS value"})

    if new_papers_per_page:
        try:
            PAPERS_PER_PAGE = int(new_papers_per_page)
        except ValueError:
            return jsonify(
                {"success": False, "message": "Invalid PAPERS_PER_PAGE value"}
            )

    if new_date_range:
        try:
            DATE_RANGE = int(new_date_range)
        except ValueError:
            return jsonify({"success": False, "message": "Invalid DATE_RANGE value"})

    if new_arxiv_categories:
        ARXIV_CATEGORIES = [cat.strip() for cat in new_arxiv_categories.split(",")]

    return jsonify({"success": True})


@app.route("/get_verdicts")
def get_verdicts():
    if os.path.exists(VERDICTS_FILE):
        with open(VERDICTS_FILE, "r") as f:
            verdicts = [json.loads(line) for line in f]
        return jsonify(verdicts)
    else:
        return jsonify([])


if __name__ == "__main__":
    app.run(debug=True)
