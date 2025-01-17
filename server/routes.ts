import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { getDatabase } from "firebase-admin/database";
import { verifyAuthToken } from "./services/firebase";
import { analyzePaperRelevance } from "./services/openai";
import { fetchAndStorePapers } from "./services/papers";

export function registerRoutes(app: Express): Server {
  // Settings endpoints
  app.get("/api/settings", requireAuth, async (req: Request, res: Response) => {
    try {
      const db = getDatabase();
      const userRef = db.ref(`users/${req.user!.uid}/preferences`);
      const snapshot = await userRef.get();
      const userPreferences = snapshot.val() || {
        preferences: "",
        categories: ["cs.LG", "cs.AI", "cs.CL"],
        openaiApiKey: null,
      };

      // Send back user preferences, masking the API key
      res.json({
        preferences: userPreferences.preferences || "",
        categories: userPreferences.categories || ["cs.LG", "cs.AI", "cs.CL"],
        openaiApiKey: userPreferences.openaiApiKey ? '********' : null,
      });
    } catch (error) {
      console.error("Error fetching settings:", error);
      res.status(500).json({ error: "Failed to fetch settings" });
    }
  });

  app.post("/api/settings", requireAuth, async (req: Request, res: Response) => {
    try {
      const { preferences, categories, openaiApiKey } = req.body;
      const db = getDatabase();
      const userRef = db.ref(`users/${req.user!.uid}/preferences`);

      // Update user preferences in Firebase
      await userRef.set({
        preferences: preferences || "",
        categories: categories || ["cs.LG", "cs.AI", "cs.CL"],
        openaiApiKey: openaiApiKey || null,
      });

      res.json({ success: true });
    } catch (error) {
      console.error("Error updating settings:", error);
      res.status(500).json({ error: "Failed to update settings" });
    }
  });

  // Papers route with intelligent search and relevance scoring
  app.get("/api/papers", requireAuth, async (req: Request, res: Response) => {
    const { preferences, page = "1", mode = "annotation" } = req.query;
    const limit = 10;
    const offset = (Number(page) - 1) * limit;

    try {
      const db = getDatabase();
      const userRef = db.ref(`users/${req.user!.uid}/preferences`);
      const snapshot = await userRef.get();
      const userPreferences = snapshot.val() || {};

      // Only require API key for relevance mode or when searching
      if ((mode === "relevance" || preferences) && !userPreferences?.openaiApiKey) {
        return res.status(400).json({
          error: "OpenAI API key required",
          requiresApiKey: true
        });
      }

      // Fetch new papers if searching
      if (preferences) {
        await fetchAndStorePapers({
          categories: userPreferences.categories || ["cs.LG", "cs.AI", "cs.CL"],
          maxResults: 100,
          dateRange: 7,
          preferences: preferences as string,
        });
      }

      // Get all papers
      const papersRef = db.ref('papers');
      const papersSnapshot = await papersRef.get();
      const allPapers = papersSnapshot.val() ? Object.values(papersSnapshot.val()) : [];

      // If searching or in relevance mode, analyze papers
      if (preferences || mode === "relevance") {
        const scoredPapers = await Promise.all(
          allPapers.map(async (paper: any) => {
            try {
              const relevance = await analyzePaperRelevance(
                paper.abstract,
                preferences as string || userPreferences.preferences || "",
                req.user!.uid
              );

              return {
                ...paper,
                relevanceScore: relevance.score,
                confidence: relevance.confidence,
                explanation: relevance.explanation
              };
            } catch (error) {
              console.error(`Error analyzing paper ${paper.id}:`, error);
              return { ...paper, relevanceScore: 0, confidence: 0 };
            }
          })
        );

        const rankedPapers = scoredPapers
          .filter(paper => paper.relevanceScore >= 50)
          .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

        const paginatedPapers = rankedPapers.slice(offset, offset + limit);
        return res.json({
          papers: paginatedPapers,
          totalPages: Math.ceil(rankedPapers.length / limit),
        });
      }

      // If not searching, return recent papers
      const paginatedPapers = allPapers.slice(offset, offset + limit);
      res.json({
        papers: paginatedPapers,
        totalPages: Math.ceil(allPapers.length / limit),
      });
    } catch (error) {
      console.error("Error processing request:", error);
      res.status(500).json({
        error: "Failed to process request",
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Vote route
  app.post("/api/papers/vote", requireAuth, async (req: Request, res: Response) => {
    const { paperId, vote } = req.body;

    try {
      const db = getDatabase();
      const voteRef = db.ref(`votes/${req.user!.uid}/${paperId}`);
      await voteRef.set({
        userId: req.user!.uid,
        paperId,
        vote: vote === 1 ? 1 : -1,
        timestamp: new Date().toISOString()
      });

      res.json({ success: true });
    } catch (error) {
      console.error("Error recording vote:", error);
      res.status(500).json({ error: "Failed to record vote" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

// Auth middleware
const requireAuth = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const token = req.headers.authorization?.split("Bearer ")[1];
    if (!token) {
      res.status(401).json({ error: "No authentication token provided" });
      return;
    }

    const decodedToken = await verifyAuthToken(token);
    req.user = decodedToken;
    next();
  } catch (error) {
    console.error("Auth error:", error);
    res.status(401).json({ error: "Authentication failed" });
  }
};