import axios from "axios";
import { parseStringPromise } from "xml2js";
import { papers, paperVotes, paperRelevanceScores } from "@db/schema";
import { db } from "@db";
import { analyzePaperRelevance, generateSearchQuery, calculatePaperSimilarity } from "./openai";
import { eq, and, desc, sql } from "drizzle-orm";

interface FetchPapersOptions {
  categories: string[];
  maxResults: number;
  dateRange: number;
  preferences?: string;
}

export async function fetchAndStorePapers(options: FetchPapersOptions) {
  const { categories, maxResults = 50, dateRange = 7, preferences } = options;

  try {
    // Generate an optimized search query if preferences are provided
    let searchQueryString: string;
    if (preferences) {
      console.log("Generating optimized search query from preferences:", preferences);
      try {
        searchQueryString = await generateSearchQuery(preferences);
        console.log("Generated search query:", searchQueryString);
      } catch (error) {
        console.error("Error generating search query:", error);
        // Fall back to category-based search
        searchQueryString = categories.map(cat => `cat:${cat}`).join("+OR+");
      }
    } else {
      searchQueryString = categories.map(cat => `cat:${cat}`).join("+OR+");
    }

    // Build the arXiv API URL with the search query and date filter
    const dateLimit = new Date();
    dateLimit.setDate(dateLimit.getDate() - dateRange);
    const dateString = dateLimit.toISOString().split('T')[0];

    const url = `http://export.arxiv.org/api/query?search_query=${searchQueryString}+AND+submittedDate:[${dateString}+TO+*]&start=0&max_results=${maxResults}&sortBy=lastUpdatedDate&sortOrder=descending`;
    console.log("Fetching papers from arXiv with URL:", url);

    const response = await axios.get(url);
    const result = await parseStringPromise(response.data, {
      explicitArray: false,
      mergeAttrs: true
    });

    const entries = Array.isArray(result.feed.entry) ? result.feed.entry : [result.feed.entry];

    console.log(`Processing ${entries.length} papers from arXiv...`);
    const results = [];
    const batchSize = 5; // Process papers in batches to improve performance

    // Process papers in batches
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      const batchPromises = batch.map(async (entry) => {
        if (!entry) return null;

        try {
          const paper = {
            arxivId: entry.id.split("/").pop()!,
            title: entry.title.replace(/\s+/g, " ").trim(),
            authors: Array.isArray(entry.author) ? 
              entry.author.map((a: any) => a.name).join(", ") : 
              entry.author.name,
            abstract: entry.summary.replace(/\s+/g, " ").trim(),
            pdfUrl: entry.id.replace("abs", "pdf"),
            abstractUrl: entry.id,
            primaryCategory: entry.primary_category ? 
              entry.primary_category.term : 
              (Array.isArray(entry.category) ? entry.category[0].term : entry.category.term),
            publishedDate: new Date(entry.published),
          };

          // If preferences are provided, analyze paper relevance before storing
          if (preferences) {
            const relevance = await analyzePaperRelevance(paper.abstract, preferences);
            console.log(`\nPaper Analysis:
              Title: ${paper.title}
              Relevance Score: ${relevance.score}
              Confidence: ${relevance.confidence}
              Abstract: ${paper.abstract.substring(0, 200)}...
              Explanation: ${relevance.explanation}
              Keywords: ${relevance.keywords?.join(", ")}
              Topic Similarity: ${relevance.topicSimilarity}
              Methodology Similarity: ${relevance.methodologySimilarity}
            `);

            // Higher threshold (70%) for more relevant results
            if (relevance.score >= 70) {
              // Check if paper already exists
              const existing = await db.select()
                .from(papers)
                .where(eq(papers.arxivId, paper.arxivId))
                .limit(1);

              if (!existing.length) {
                const [storedPaper] = await db.insert(papers)
                  .values(paper)
                  .returning();

                // Store relevance scores for better recommendations
                await db.insert(paperRelevanceScores).values({
                  paperId: storedPaper.id,
                  score: relevance.score,
                  confidence: relevance.confidence,
                  modelResponse: relevance
                });

                return { ...paper, relevance };
              }
            } else {
              console.log(`Skipping paper ${paper.arxivId} due to low relevance score: ${relevance.score}`);
            }
          } else {
            // Without preferences, store all papers
            const existing = await db.select()
              .from(papers)
              .where(eq(papers.arxivId, paper.arxivId))
              .limit(1);

            if (!existing.length) {
              await db.insert(papers).values(paper);
              return paper;
            }
          }
        } catch (error) {
          console.error(`Failed to process paper:`, error);
        }
        return null;
      });

      const batchResults = await Promise.all(batchPromises);
      const validResults = batchResults.filter(Boolean);

      // Sort batch by relevance score before adding to results
      const sortedResults = validResults.sort((a: any, b: any) => 
        (b.relevance?.score || 0) - (a.relevance?.score || 0)
      );

      results.push(...sortedResults);

      console.log(`Processed batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(entries.length/batchSize)}`);
    }

    // Log top 20 papers
    console.log("\nTop 20 Papers by Relevance:");
    results.slice(0, 20).forEach((paper: any, index: number) => {
      console.log(`\n#${index + 1}:
        Title: ${paper.title}
        Score: ${paper.relevance?.score}
        Confidence: ${paper.relevance?.confidence}
        Explanation: ${paper.relevance?.explanation}
        Abstract Preview: ${paper.abstract.substring(0, 100)}...`);
    });

    console.log(`\nSuccessfully processed ${results.length} new papers`);
    return results.slice(0, maxResults); // Return only the top papers up to maxResults
  } catch (error) {
    console.error("Error fetching papers from arXiv:", error);
    if ((error as any).response?.data) {
      console.error("arXiv API response:", (error as any).response.data);
    }
    return [];
  }
}

export async function getPaperRelevance(paperId: number, preferences: string) {
  const [paper] = await db.select()
    .from(papers)
    .where(eq(papers.id, paperId))
    .limit(1);

  if (!paper) {
    throw new Error("Paper not found");
  }

  const relevance = await analyzePaperRelevance(paper.abstract, preferences);
  return relevance;
}

export async function updateRelevanceScores(userId: number, preferences: string) {
  // Get all papers for this user
  const userPapers = await db.select()
    .from(papers)
    .innerJoin(paperVotes, eq(papers.id, paperVotes.paperId))
    .where(eq(paperVotes.userId, userId));

  // Update relevance scores based on preferences and voting history
  for (const paper of userPapers) {
    try {
      const relevance = await analyzePaperRelevance(paper.abstract, preferences);

      await db.insert(paperRelevanceScores)
        .values({
          paperId: paper.id,
          userId,
          score: relevance.score,
          confidence: relevance.confidence,
          modelResponse: relevance
        })
        .onConflictDoUpdate({
          target: [paperRelevanceScores.paperId, paperRelevanceScores.userId],
          set: {
            score: relevance.score,
            confidence: relevance.confidence,
            modelResponse: relevance
          }
        });
    } catch (error) {
      console.error(`Error updating relevance score for paper ${paper.id}:`, error);
    }
  }
}