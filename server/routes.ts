import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { verifyAuthToken } from "./services/firebase";
import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql } from "drizzle-orm";
import { analyzePaperRelevance } from "./services/openai";
import { fetchAndStorePapers } from "./services/papers";

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

    // Check if user exists in our database
    const [existingUser] = await db.select()
      .from(users)
      .where(eq(users.firebaseId, decodedToken.uid));

    if (!existingUser) {
      // Create new user if they don't exist
      await db.insert(users).values({
        firebaseId: decodedToken.uid,
        email: decodedToken.email!,
      });
    }

    next();
  } catch (error) {
    console.error("Auth error:", error);
    res.status(401).json({ error: "Authentication failed" });
  }
};

export function registerRoutes(app: Express): Server {
  // Environment variables route for client
  app.get("/api/config", (_req, res) => {
    res.json({
      firebaseApiKey: process.env.FIREBASE_API_KEY,
      firebaseProjectId: process.env.FIREBASE_PROJECT_ID,
      firebaseAppId: process.env.FIREBASE_APP_ID,
    });
  });

  // Papers route with intelligent search and relevance scoring
  app.get("/api/papers", requireAuth, async (req: Request, res: Response) => {
    const { preferences = "", page = "1", mode = "relevance" } = req.query;
    const limit = 10;
    const offset = (Number(page) - 1) * limit;

    try {
      // Get the user's ID
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      // Fetch papers using intelligent search if preferences are provided
      const existingPaperCount = await db.select({ count: sql<number>`count(*)` })
        .from(papers)
        .execute();

      if (existingPaperCount[0].count < limit * 2 || preferences) {
        console.log("Fetching papers with preferences:", preferences);
        await fetchAndStorePapers({
          categories: ["cs.LG", "cs.AI", "cs.CL"],
          maxResults: 100,
          dateRange: 7,
          preferences: preferences as string,
        });
      }

      // Base query for papers
      let query = db.select()
        .from(papers)
        .orderBy(desc(papers.publishedDate));

      // Modify query based on mode
      if (mode === "annotation") {
        const votedPapers = await db.select()
          .from(paperVotes)
          .where(eq(paperVotes.userId, user.id));

        if (votedPapers.length > 0) {
          const votedIds = votedPapers.map(v => v.paperId);
          query = query.where(sql`${papers.id} NOT IN (${votedIds.join(",")})`);
        }
      }

      // Get papers and analyze relevance
      const allPapers = await query.execute();
      let processedPapers = allPapers;

      if (mode === "relevance" && preferences) {
        console.log("Analyzing paper relevance for preferences:", preferences);
        const scoredPapers = await Promise.all(
          allPapers.map(async (paper) => {
            try {
              const relevance = await analyzePaperRelevance(paper.abstract, preferences as string);
              return {
                ...paper,
                relevanceScore: relevance.score,
                confidence: relevance.confidence,
              };
            } catch (error) {
              console.error(`Error analyzing paper ${paper.id}:`, error);
              return {
                ...paper,
                relevanceScore: 0,
                confidence: 0,
              };
            }
          })
        );

        // Sort by relevance score and filter out low-relevance papers
        processedPapers = scoredPapers
          .filter(paper => paper.relevanceScore >= 50)
          .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
      }

      // Paginate results
      const paginatedPapers = processedPapers.slice(offset, offset + limit);

      res.json({
        papers: paginatedPapers,
        totalPages: Math.ceil(processedPapers.length / limit),
      });
    } catch (error) {
      console.error("Error processing papers request:", error);
      res.status(500).json({ 
        error: "Failed to process request", 
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Vote route
  app.post("/api/papers/vote", requireAuth, async (req: Request, res: Response) => {
    const { paperId, vote } = req.body;

    try {
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      await db.insert(paperVotes).values({
        paperId,
        userId: user.id,
        vote: vote === 1 ? 1 : -1,
      });

      res.json({ success: true });
    } catch (error) {
      console.error("Error recording vote:", error);
      res.status(500).json({ error: "Failed to record vote" });
    }
  });

  // Metrics route
  app.get("/api/metrics", requireAuth, async (req: Request, res: Response) => {
    try {
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      const votes = await db.select()
        .from(paperVotes)
        .where(eq(paperVotes.userId, user.id));

      const metrics = {
        totalVotes: votes.length,
        upvotes: votes.filter(v => v.vote === 1).length,
        downvotes: votes.filter(v => v.vote === -1).length,
        averageRelevanceScore: 0,
        averageConfidence: 0,
      };

      res.json(metrics);
    } catch (error) {
      console.error("Error fetching metrics:", error);
      res.status(500).json({ error: "Failed to fetch metrics" });
    }
  });

  // Settings route
  app.post("/api/settings", requireAuth, async (req: Request, res: Response) => {
    const { preferences, categories } = req.body;

    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      await db.update(users)
        .set({ preferences: { preferences, categories } })
        .where(eq(users.id, user.id));

      res.json({ success: true });
    } catch (error) {
      console.error("Error saving settings:", error);
      res.status(500).json({ error: "Failed to save settings" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}