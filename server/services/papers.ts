import { search } from "arxiv";
import { papers } from "@db/schema";
import { db } from "@db";
import { analyzePaperRelevance } from "../services/openai";
import { eq } from "drizzle-orm";

interface FetchPapersOptions {
  categories: string[];
  maxResults: number;
  dateRange: number;
}

export async function fetchAndStorePapers(options: FetchPapersOptions) {
  const { categories, maxResults, dateRange } = options;

  try {
    // Build the category query string
    const categoryQuery = categories.map(cat => `cat:${cat}`).join(" OR ");

    // Use the arxiv package correctly
    const searchResults = await search({
      searchQuery: categoryQuery,
      start: 0,
      maxResults: maxResults,
      sortBy: 'lastUpdatedDate',
      sortOrder: 'descending'
    }).catch(error => {
      console.error("ArXiv search error:", error);
      return [];
    });

    if (!Array.isArray(searchResults)) {
      console.warn("No results returned from arXiv");
      return [];
    }

    const dateLimit = new Date();
    dateLimit.setDate(dateLimit.getDate() - dateRange);

    const results = [];
    for (const result of searchResults) {
      const published = new Date(result.published);
      if (published >= dateLimit) {
        try {
          const paper = {
            arxivId: result.id.split("/").pop()!,
            title: result.title,
            authors: result.authors.join(", "),
            abstract: result.summary,
            pdfUrl: result.links.find(link => link.includes("pdf"))!,
            abstractUrl: result.id,
            primaryCategory: result.categories[0],
            publishedDate: published,
          };

          const existing = await db.select()
            .from(papers)
            .where(eq(papers.arxivId, paper.arxivId));

          if (!existing.length) {
            await db.insert(papers).values(paper);
            results.push(paper);
          }
        } catch (error) {
          console.error(`Failed to process paper:`, error);
          continue;
        }
      }
    }

    return results;
  } catch (error) {
    console.error("Error fetching papers from arXiv:", error);
    return [];
  }
}

export async function getPaperRelevance(paperId: number, preferences: string) {
  const paper = await db.select().from(papers).where(eq(papers.id, paperId)).limit(1);

  if (!paper.length) {
    throw new Error("Paper not found");
  }

  const relevance = await analyzePaperRelevance(paper[0].abstract, preferences);
  return relevance;
}