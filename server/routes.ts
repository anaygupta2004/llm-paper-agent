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

  // Papers route
  app.get("/api/papers", requireAuth, async (req: Request, res: Response) => {
    const { preferences = "", page = "1", mode = "relevance" } = req.query;
    const limit = 10;
    const offset = (Number(page) - 1) * limit;

    try {
      // Get the user's ID
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      // Fetch fresh papers from arXiv if we're running low
      const existingPapersCount = await db.select({ count: sql<number>`count(*)` })
        .from(papers)
        .execute();

      if (existingPapersCount[0].count < limit * 2) {
        try {
          await fetchAndStorePapers({
            categories: ["cs.LG", "cs.AI", "cs.CL"], // Default categories
            maxResults: 100,
            dateRange: 7,
          });
        } catch (error) {
          console.error("Error fetching new papers:", error);
          // Continue with existing papers if fetch fails
        }
      }

      // Get papers that haven't been voted on for annotation mode
      let query = db.select()
        .from(papers)
        .orderBy(desc(papers.publishedDate));

      if (mode === "annotation") {
        const votedPaperIds = await db.select()
          .from(paperVotes)
          .where(eq(paperVotes.userId, user.id));

        if (votedPaperIds.length > 0) {
          query = query.where(
            sql`${papers.id} NOT IN ${votedPaperIds.map(v => v.paperId)}`
          );
        }
      }

      // Get all available papers
      const allPapers = await query.execute();

      // Apply relevance scoring if needed
      let relevantPapers = allPapers;
      if (mode === "relevance" && preferences) {
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

        // Sort by relevance score for relevance mode
        relevantPapers = scoredPapers.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
      }

      // Paginate results
      const paginatedPapers = relevantPapers.slice(offset, offset + limit);

      res.json({
        papers: paginatedPapers,
        totalPages: Math.ceil(relevantPapers.length / limit),
      });
    } catch (error) {
      console.error("Error processing papers request:", error);
      res.status(500).json({ error: "Failed to process request" });
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