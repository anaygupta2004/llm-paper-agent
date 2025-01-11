import axios from "axios";
import { parseStringPromise } from "xml2js";
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
    const categoryQuery = categories.map(cat => `cat:${cat}`).join("+OR+");
    const url = `http://export.arxiv.org/api/query?search_query=${categoryQuery}&start=0&max_results=${maxResults}&sortBy=lastUpdatedDate&sortOrder=descending`;

    const response = await axios.get(url);
    const result = await parseStringPromise(response.data, {
      explicitArray: false,
      mergeAttrs: true
    });

    const entries = Array.isArray(result.feed.entry) ? result.feed.entry : [result.feed.entry];
    const dateLimit = new Date();
    dateLimit.setDate(dateLimit.getDate() - dateRange);

    const results = [];
    for (const entry of entries) {
      if (!entry) continue;

      const published = new Date(entry.published);
      if (published >= dateLimit) {
        try {
          const paper = {
            arxivId: entry.id.split("/").pop()!,
            title: entry.title.replace(/\s+/g, " ").trim(),
            authors: Array.isArray(entry.author) ? 
              entry.author.map((a: any) => a.name).join(", ") : 
              entry.author.name,
            abstract: entry.summary.replace(/\s+/g, " ").trim(),
            pdfUrl: `${entry.id.replace("abs", "pdf")}`,
            abstractUrl: entry.id,
            primaryCategory: entry.primary_category ? 
              entry.primary_category.term : 
              (Array.isArray(entry.category) ? entry.category[0].term : entry.category.term),
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