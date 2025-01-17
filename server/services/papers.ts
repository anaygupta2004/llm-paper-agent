import axios from "axios";
import { parseStringPromise } from "xml2js";
import { getDatabase, type Database } from "firebase-admin/database";
import { analyzePaperRelevance, generateSearchQuery } from "./openai";

interface FetchPapersOptions {
  categories: string[];
  maxResults: number;
  dateRange: number;
  preferences?: string;
  userId?: string;
}

export async function fetchAndStorePapers(options: FetchPapersOptions) {
  const { categories, maxResults = 50, dateRange = 7, preferences, userId } = options;

  try {
    console.log("\n========== STARTING PAPER SEARCH ==========");
    console.log(`Search Parameters:
    - Categories: ${categories.join(', ')}
    - Max Results: ${maxResults}
    - Date Range: ${dateRange} days
    - Preferences: ${preferences || 'None'}`);

    // Generate an optimized search query if preferences are provided
    let searchQueryString: string;
    if (preferences) {
      console.log("\n========== GENERATING SEARCH QUERY ==========");
      try {
        searchQueryString = await generateSearchQuery(preferences, userId);
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
    console.log("\n========== FETCHING FROM ARXIV ==========");
    console.log("URL:", url);

    const response = await axios.get(url);
    const result = await parseStringPromise(response.data, {
      explicitArray: false,
      mergeAttrs: true
    });

    const entries = Array.isArray(result.feed.entry) ? result.feed.entry : [result.feed.entry];

    console.log("\n========== PROCESSING PAPERS ==========");
    console.log(`Found ${entries.length} papers to process`);

    const db = getDatabase();
    const results = [];
    const batchSize = 5; // Process papers in batches to improve performance

    // Process papers in batches
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      console.log(`\nProcessing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(entries.length/batchSize)}`);

      const batchPromises = batch.map(async (entry: any) => {
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
            publishedDate: new Date(entry.published).toISOString(),
          };

          // If preferences are provided, analyze paper relevance before storing
          if (preferences) {
            const relevance = await analyzePaperRelevance(paper.abstract, preferences, userId);
            console.log(`\n----- Paper Analysis -----
Title: ${paper.title}
Relevance Score: ${relevance.score}%
Confidence: ${relevance.confidence}%
Keywords: ${relevance.keywords?.join(", ")}
Topic Similarity: ${relevance.topicSimilarity}%
Methodology Similarity: ${relevance.methodologySimilarity}%
Explanation: ${relevance.explanation}
Abstract: ${paper.abstract.substring(0, 200)}...
`);

            // Higher threshold (70%) for more relevant results
            if (relevance.score >= 70) {
              // Check if paper already exists
              const paperRef = db.ref(`papers/${paper.arxivId}`);
              const snapshot = await paperRef.get();

              if (!snapshot.exists()) {
                // Store paper in Firebase
                await paperRef.set(paper);

                // Store relevance scores for better recommendations
                if (userId) {
                  await db.ref(`paperRelevanceScores/${userId}/${paper.arxivId}`).set({
                    score: relevance.score,
                    confidence: relevance.confidence,
                    modelResponse: relevance,
                    createdAt: new Date().toISOString()
                  });
                }

                return { ...paper, relevance };
              }
            } else {
              console.log(`\nSkipping paper ${paper.arxivId} - Low relevance score: ${relevance.score}`);
            }
          } else {
            // Without preferences, store all papers
            const paperRef = db.ref(`papers/${paper.arxivId}`);
            const snapshot = await paperRef.get();

            if (!snapshot.exists()) {
              await paperRef.set(paper);
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
    }

    console.log("\n========== TOP 20 PAPERS BY RELEVANCE ==========");
    results.slice(0, 20).forEach((paper: any, index: number) => {
      console.log(`\n#${index + 1}. ${paper.title}
Score: ${paper.relevance?.score}%
Confidence: ${paper.relevance?.confidence}%
Explanation: ${paper.relevance?.explanation}
Abstract: ${paper.abstract.substring(0, 150)}...
----------------------------------------`);
    });

    console.log(`\n========== SEARCH COMPLETED ==========`);
    console.log(`Successfully processed ${results.length} papers`);

    return results.slice(0, maxResults); // Return only the top papers up to maxResults
  } catch (error) {
    console.error("\n========== ERROR ==========");
    console.error("Error fetching papers from arXiv:", error);
    if ((error as any).response?.data) {
      console.error("arXiv API response:", (error as any).response.data);
    }
    return [];
  }
}

export async function getPaperRelevance(paperId: string, preferences: string, userId?: string) {
  const db = getDatabase();
  const paperRef = db.ref(`papers/${paperId}`);
  const snapshot = await paperRef.get();

  if (!snapshot.exists()) {
    throw new Error("Paper not found");
  }

  const paper = snapshot.val();
  const relevance = await analyzePaperRelevance(paper.abstract, preferences, userId);
  return relevance;
}

export async function updateRelevanceScores(userId: string, preferences: string) {
  const db = getDatabase();

  // Get all papers for this user
  const votesRef = db.ref(`votes/${userId}`);
  const votesSnapshot = await votesRef.get();
  const votes = votesSnapshot.val() || {};

  // Get papers
  const papersRef = db.ref('papers');
  const papersSnapshot = await papersRef.get();
  const papers = papersSnapshot.val() || {};

  // Update relevance scores based on preferences and voting history
  for (const paperId of Object.keys(votes)) {
    try {
      const paper = papers[paperId];
      if (!paper) continue;

      const relevance = await analyzePaperRelevance(paper.abstract, preferences, userId);

      await db.ref(`paperRelevanceScores/${userId}/${paperId}`).set({
        score: relevance.score,
        confidence: relevance.confidence,
        modelResponse: relevance,
        updatedAt: new Date().toISOString()
      });
    } catch (error) {
      console.error(`Error updating relevance score for paper ${paperId}:`, error);
    }
  }
}