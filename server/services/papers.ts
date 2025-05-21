import axios from "axios";
import { parseStringPromise } from "xml2js";
import { getDatabase } from "firebase-admin/database";
import { analyzePaperRelevance, generateSearchQuery } from "./openai";

// Function to sanitize paper IDs for Firebase paths
// Firebase doesn't allow '.', '#', '$', '[', or ']' in path segments
function sanitizeForFirebasePath(id: string): string {
  return id.replace(/[.#$[\]]/g, '_');
}

interface FetchPapersOptions {
  categories: string[];
  maxResults: number;
  dateRange: number; // in days
  preferences?: string;
  userId?: string;
  apiKey?: string;
}

export async function fetchAndStorePapers(options: FetchPapersOptions) {
  const { categories, maxResults = 50, dateRange = 50, preferences, userId, apiKey } = options;

  try {
    console.log("\n========== STARTING PAPER SEARCH ==========");
    console.log(`Search Parameters:
    - Categories: ${categories.join(', ')}
    - Max Results: ${maxResults}
    - Date Range: ${dateRange} days
    - Preferences: ${preferences || 'None'}`);

    // Generate an optimized search query if preferences provided,
    // otherwise fall back to categories.
    let searchQueryString: string;
    if (preferences) {
      console.log("\n========== GENERATING SEARCH QUERY VIA OPENAI ==========");
      try {
        searchQueryString = await generateSearchQuery(preferences, userId, apiKey);
        if (!searchQueryString || searchQueryString.trim() === "") {
          console.warn("OpenAI generated an empty search query. Falling back to categories.");
          throw new Error("Empty query from OpenAI"); // Force fallback
        }
        console.log("Generated search query (from OpenAI):", searchQueryString);
      } catch (error) {
        console.error("Error generating search query via OpenAI:", error);
        console.log("Falling back to category-based search for ArXiv query.");
        if (categories && categories.length > 0) {
          searchQueryString = categories.map(cat => `cat:${cat}`).join(" OR "); // Changed +OR+ to OR with spaces
        } else {
          searchQueryString = ""; // Explicitly empty if no categories
        }
        console.log("Fallback search query (from categories):", searchQueryString);
      }
    } else {
      console.log("\n========== USING CATEGORY-BASED SEARCH QUERY ==========");
      if (categories && categories.length > 0) {
        searchQueryString = categories.map(cat => `cat:${cat}`).join(" OR "); // Changed +OR+ to OR with spaces
      } else {
        searchQueryString = ""; // Explicitly empty if no categories
      }
      console.log("Category-based search query:", searchQueryString);
    }

    if (!searchQueryString || searchQueryString.trim() === "") {
      console.error("Search query string is empty. Cannot fetch from ArXiv. Please check preferences or categories.");
      return []; // Return empty if no valid query can be formed
    }

    // Calculate date limit for the query
    const dateLimit = new Date();
    dateLimit.setDate(dateLimit.getDate() - dateRange);
    const dateString = dateLimit.toISOString().slice(0, 10).replace(/-/g, '') + '000000';

    // Construct the full ArXiv query string
    // If searchQueryString already seems to be a grouped expression, don't add extra parentheses.
    // Fix the query format:
    // 1. Replace any instances of 'title:' with 'ti:' in the search query to match ArXiv's format
    // 2. Remove quotes around search terms that make ArXiv API unhappy
    let queryPart = searchQueryString.replace(/title:/g, 'ti:').replace(/["']/g, '');
    if (!(queryPart.startsWith("(") && queryPart.endsWith(")"))) {
      queryPart = `(${queryPart})`;
    }
    const fullArxivQuery = `${queryPart} AND lastUpdatedDate:[${dateString} TO *]`;

    const arxivApiUrl = `http://export.arxiv.org/api/query`;
    const arxivParams = {
        search_query: fullArxivQuery,
        start: 0,
        max_results: maxResults,
        sortBy: "lastUpdatedDate",
        sortOrder: "descending"
    };
    
    console.log("\n========== FETCHING FROM ARXIV ==========");
    console.log("Raw search query (from OpenAI/categories):", searchQueryString);
    console.log("Date limit for ArXiv query:", dateString);
    console.log("Constructed ArXiv query (pre-axios encoding):", fullArxivQuery);
    console.log("ArXiv API URL (base):", arxivApiUrl);
    console.log("Params for axios:", arxivParams);

    // Let axios handle the URL encoding of the parameters
    const response = await axios.get(arxivApiUrl, { params: arxivParams });
    console.log("ArXiv API Response Status:", response.status);
    console.log("ArXiv API Response Headers:", response.headers);
    console.log("ArXiv API Response Data:", response.data);
    
    const result = await parseStringPromise(response.data, {
      explicitArray: false,
      mergeAttrs: true
    });

    // Extract entries (ensure we always have an array)
    let entries: any[] = result.feed && result.feed.entry
      ? Array.isArray(result.feed.entry)
        ? result.feed.entry
        : [result.feed.entry]
      : [];

    console.log("Number of entries found with date filter:", entries.length);
    if (entries.length === 0) {
      console.warn("No papers found in date range. Retrying without date filter...");
      // Retry without date filter
      const fallbackParams = {
        search_query: queryPart,
        start: 0,
        max_results: maxResults,
        sortBy: "lastUpdatedDate",
        sortOrder: "descending"
      };
      const fallbackResponse = await axios.get(arxivApiUrl, { params: fallbackParams });
      console.log("Fallback ArXiv API Response Status:", fallbackResponse.status);
      console.log("Fallback ArXiv API Response Data:", fallbackResponse.data);
      const fallbackResult = await parseStringPromise(fallbackResponse.data, { explicitArray: false, mergeAttrs: true });
      entries = fallbackResult.feed && fallbackResult.feed.entry
        ? Array.isArray(fallbackResult.feed.entry)
          ? fallbackResult.feed.entry
          : [fallbackResult.feed.entry]
        : [];
      console.log("Number of entries found without date filter:", entries.length);
      if (entries.length === 0) {
        console.log("Still no papers found from ArXiv.");
        return [];
      }
      console.log("Using results without date filter.");
    }

    // No need for manual date filtering since we're using the API's date filter (or have retried without it)
    const filteredEntries = entries;

    const database = getDatabase();
    const resultsArray: any[] = [];
    const batchSize = 5; // Process papers in batches

    // Process in batches for performance.
    for (let i = 0; i < filteredEntries.length; i += batchSize) {
      const batch = filteredEntries.slice(i, i + batchSize);
      console.log(`\nProcessing batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(filteredEntries.length / batchSize)}`);

      const batchPromises = batch.map(async (entry: any) => {
        if (!entry || !entry.id) return null;
        try {
          // Build paper object
          const paper = {
            arxivId: entry.id.split("/").pop()!,
            title: (entry.title || "").replace(/\s+/g, " ").trim(),
            authors: Array.isArray(entry.author)
              ? entry.author.map((a: any) => a.name).join(", ")
              : entry.author?.name || "Unknown",
            abstract: (entry.summary || "").replace(/\s+/g, " ").trim(),
            pdfUrl: entry.id.replace("abs", "pdf"),
            abstractUrl: entry.id,
            primaryCategory: entry.primary_category
              ? entry.primary_category.term
              : (Array.isArray(entry.category)
                ? entry.category[0].term
                : entry.category?.term || "Unknown"),
            publishedDate: new Date(entry.published).toISOString(),
          };

          // If preferences are provided, check relevance before storing.
          if (preferences) {
            if (!apiKey) {
              console.error("Missing OpenAI API key for paper analysis");
              throw new Error("API key required. Please set your OpenAI API key in settings to enable paper analysis.");
            }

            try {
              const relevance = await analyzePaperRelevance(paper.abstract, preferences, userId, apiKey);
              console.log(`\n----- Paper Analysis -----
Title: ${paper.title}
Relevance Score: ${relevance.score}%
Confidence: ${relevance.confidence}%
Keywords: ${relevance.keywords?.join(", ")}
Topic Similarity: ${relevance.topicSimilarity}%
Methodology Similarity: ${relevance.methodologySimilarity}%
Explanation: ${relevance.explanation}
Abstract (first 200 chars): ${paper.abstract.substring(0, 200)}...
`);

              // Only store papers with a high relevance score
              if (relevance.score >= 70) {
                const sanitizedId = sanitizeForFirebasePath(paper.arxivId);
                const paperRef = database.ref(`papers/${sanitizedId}`);
                const snapshot = await paperRef.get();
                if (!snapshot.exists()) {
                  await paperRef.set(paper);
                  // If userId provided, store additional relevance scores.
                  if (userId) {
                    await database.ref(`paperRelevanceScores/${userId}/${sanitizedId}`).set({
                      score: relevance.score,
                      confidence: relevance.confidence,
                      modelResponse: relevance,
                      createdAt: new Date().toISOString()
                    });
                  }
                  return { ...paper, relevance };
                } else {
                  // Return the already-stored paper.
                  return { ...paper, relevance };
                }
              } else {
                console.log(`\nSkipping paper ${paper.arxivId} - Low relevance score: ${relevance.score}`);
                return null;
              }
            } catch (error: any) {
              console.error(`Error analyzing paper ${paper.arxivId}:`, error);
              
              // Check for API key related errors
              if (error.message && (
                  error.message.includes("API key") || 
                  error.message.includes("authentication") ||
                  error.message.includes("auth") ||
                  error.message.includes("Invalid key")
              )) {
                throw new Error("API key required or invalid. Please check your OpenAI API key in settings to enable paper analysis.");
              }
              
              throw error; // Propagate other errors to be handled by the caller
            }
          } else {
            // If no preferences, store every paper that is not already in DB.
            const sanitizedId = sanitizeForFirebasePath(paper.arxivId);
            const paperRef = database.ref(`papers/${sanitizedId}`);
            const snapshot = await paperRef.get();
            if (!snapshot.exists()) {
              await paperRef.set(paper);
            }
            return paper;
          }
        } catch (error) {
          console.error(`Failed to process paper:`, error);
          return null;
        }
      });

      const batchResults = await Promise.all(batchPromises);
      // Filter out nulls and sort (if relevance info is available, else default to 0)
      const validResults = batchResults.filter(result => result !== null);
      const sortedResults = validResults.sort((a: any, b: any) =>
        ((b.relevance?.score ?? 0) - (a.relevance?.score ?? 0))
      );
      resultsArray.push(...sortedResults);
    }

    console.log("\n========== TOP 20 PAPERS BY RELEVANCE ==========");
    resultsArray.slice(0, 20).forEach((paper: any, index: number) => {
      console.log(`\n#${index + 1}. ${paper.title}
Score: ${paper.relevance?.score || "N/A"}%
Confidence: ${paper.relevance?.confidence || "N/A"}%
Explanation: ${paper.relevance?.explanation || "No explanation provided"}
Abstract (first 150 chars): ${paper.abstract.substring(0, 150)}...
----------------------------------------`);
    });

    console.log(`\n========== SEARCH COMPLETED ==========`);
    console.log(`Successfully processed ${resultsArray.length} papers`);
    return resultsArray.slice(0, maxResults); // Return up to maxResults papers
  } catch (error) {
    console.error("\n========== ERROR ==========");
    console.error("Error fetching papers from arXiv:", error);
    if ((error as any).response?.data) {
      console.error("arXiv API response:", (error as any).response.data);
    }
    return [];
  }
}

export async function getPaperRelevance(paperId: string, preferences: string, userId?: string, apiKey?: string) {
  const database = getDatabase();
  const sanitizedId = sanitizeForFirebasePath(paperId);
  const paperRef = database.ref(`papers/${sanitizedId}`);
  const snapshot = await paperRef.get();
  if (!snapshot.exists()) {
    throw new Error("Paper not found");
  }
  const paper = snapshot.val();
  const relevance = await analyzePaperRelevance(paper.abstract, preferences, userId, apiKey);
  return relevance;
}

export async function updateRelevanceScores(userId: string, preferences: string, apiKey?: string) {
  const database = getDatabase();
  // Get all votes for this user (assumes votes are stored per user)
  const votesRef = database.ref(`votes/${userId}`);
  const votesSnapshot = await votesRef.get();
  const votes = votesSnapshot.val() || {};

  // Get all stored papers
  const papersRef = database.ref('papers');
  const papersSnapshot = await papersRef.get();
  const papers = papersSnapshot.val() || {};

  // Update relevance scores for each paper the user has voted on.
  for (const paperId of Object.keys(votes)) {
    try {
      const sanitizedId = sanitizeForFirebasePath(paperId);
      const paper = papers[sanitizedId];
      if (!paper) continue;
      const relevance = await analyzePaperRelevance(paper.abstract, preferences, userId, apiKey);
      await database.ref(`paperRelevanceScores/${userId}/${sanitizedId}`).set({
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
