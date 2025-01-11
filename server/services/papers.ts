import arxiv from "arxiv";
import { papers } from "@db/schema";
import { db } from "@db";
import { analyzePaperRelevance } from "./openai";
import { eq } from "drizzle-orm";

interface FetchPapersOptions {
  categories: string[];
  maxResults: number;
  dateRange: number;
}

export async function fetchAndStorePapers(options: FetchPapersOptions) {
  const { categories, maxResults, dateRange } = options;

  const search = arxiv.Search({
    query: categories.map(cat => `cat:${cat}`).join(" OR "),
    maxResults,
    sortBy: arxiv.SortCriterion.SubmittedDate,
  });

  const client = arxiv.Client();
  const dateLimit = new Date();
  dateLimit.setDate(dateLimit.getDate() - dateRange);

  const results = [];
  for await (const result of client.results(search)) {
    if (result.published >= dateLimit) {
      const paper = {
        arxivId: result.entry_id.split("/").pop()!,
        title: result.title,
        authors: result.authors.join(", "),
        abstract: result.summary,
        pdfUrl: result.pdf_url,
        abstractUrl: result.entry_id,
        primaryCategory: result.primary_category,
        publishedDate: result.published,
      };

      try {
        const existing = await db.select().from(papers).where(eq(papers.arxivId, paper.arxivId));
        
        if (!existing.length) {
          await db.insert(papers).values(paper);
          results.push(paper);
        }
      } catch (error) {
        console.error(`Failed to store paper ${paper.arxivId}:`, error);
      }
    }
  }

  return results;
}

export async function getPaperRelevance(paperId: number, preferences: string) {
  const paper = await db.select().from(papers).where(eq(papers.id, paperId)).limit(1);
  
  if (!paper.length) {
    throw new Error("Paper not found");
  }

  const relevance = await analyzePaperRelevance(paper[0].abstract, preferences);
  return relevance;
}
