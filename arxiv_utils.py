import arxiv
import datetime
from datetime import datetime, timedelta, timezone
import json
import re

from gptquery import GPT


def fetch_recent_papers(arxiv_categories,
                        max_results,
                        date_range):
    search = arxiv.Search(
        query=" OR ".join(f"cat:{cat}" for cat in arxiv_categories),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    papers = []
    date_limit = datetime.now(timezone.utc) - timedelta(days=date_range)

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


def evaluate_relevance(paper, 
                       preferences,
                       model_name,
                       key,
                       verdicts_file,):
    evaluation_prompt = """\
You are an AI assistant helping an AI researcher go through relevant papers.
The researcher has the following preferences: {preferences}
Now consider the following article:
Title: {title}
Abstract: {abstract}
Is this paper strongly relevant? If so reply with RELEVANT, if not reply NOT_ENOUGH_RELATED. \
Enclose your response in the tag <decision>RELEVANT/NOT_ENOUGH_RELATED</decision>
"""
    try:
        llm = GPT(
            model_name=model_name,
            task_prompt_text=evaluation_prompt,
            oai_key=key,
            max_num_tokens=128,
        )

        input = {
            **paper,
            "preferences": preferences,
        }
        result = llm([input])[0]
        model_verdict = "RELEVANT" in re.findall(r"<decision>([\w\W]*)</decision>", result["response"])

        # Append the verdict to the file
        with open(verdicts_file, "a") as f:
            json.dump(
                {
                    **paper,
                    **{
                        "preferences": preferences,
                        "model_verdict": model_verdict,
                    }
                },
                f,
            )
            f.write("\n")

        return model_verdict
    except Exception as e:
        print(f"Error in evaluate_relevance: {str(e)}")
        return False


if __name__ == "__main__":
    papers = fetch_recent_papers(["cs.LG"],
                                200,
                                4)
    print(len(papers))