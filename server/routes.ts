import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { analyzePaperRelevance } from "./services/openai";
import { getRecommendations, updateRecommendations } from "./services/recommendations";
import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql } from "drizzle-orm";
import { verifyAuthToken } from "./services/firebase";

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
    const [existingUser] = await db.select().from(users).where(eq(users.firebaseId, decodedToken.uid));

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
  // Paper routes
  app.get("/api/papers", requireAuth, async (req: Request, res: Response) => {
    const { preferences = "", page = "1", mode = "relevance" } = req.query;
    const limit = 10;
    const offset = (Number(page) - 1) * limit;

    try {
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      if (mode === "annotation") {
        // In annotation mode, show papers that need feedback
        const allPapers = await db.query.papers.findMany({
          orderBy: [desc(papers.publishedDate)],
          limit,
          offset
        });

        const recommendations = await getRecommendations(user.id, preferences as string);
        const needFeedbackPapers = allPapers.filter(paper => 
          recommendations.find(r => r.paperId === paper.id && r.needsFeedback)
        );

        const totalCount = await db.select({ count: sql`count(*)` }).from(papers);
        const totalPages = Math.ceil(totalCount[0].count / limit);

        res.json({
          papers: needFeedbackPapers,
          totalPages,
          mode: "annotation"
        });
      } else {
        // In relevance mode, show the most relevant papers
        const recommendations = await getRecommendations(user.id, preferences as string);

        // Sort by score and get the page
        const sortedRecommendations = recommendations
          .sort((a, b) => b.score - a.score)
          .slice(offset, offset + limit);

        // Get the actual paper data
        const recommendedPapers = await Promise.all(
          sortedRecommendations.map(async (rec) => {
            const [paper] = await db.select()
              .from(papers)
              .where(eq(papers.id, rec.paperId));

            return {
              ...paper,
              relevanceScore: rec.score,
              confidence: rec.confidence
            };
          })
        );

        const totalPages = Math.ceil(recommendations.length / limit);

        res.json({
          papers: recommendedPapers,
          totalPages,
          mode: "relevance"
        });
      }
    } catch (error) {
      console.error("Error fetching papers:", error);
      res.status(500).json({ error: "Failed to fetch papers" });
    }
  });

  app.post("/api/papers/vote", requireAuth, async (req: Request, res: Response) => {
    const { paperId, vote } = req.body;

    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      // Record the vote
      await db.insert(paperVotes).values({
        paperId,
        userId: user.id,
        vote
      });

      // Update recommendations based on feedback
      await updateRecommendations(user.id, paperId, vote);

      res.json({ success: true });
    } catch (error) {
      console.error("Error recording vote:", error);
      res.status(500).json({ error: "Failed to record vote" });
    }
  });

  app.get("/api/metrics", requireAuth, async (req: Request, res: Response) => {
    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      const userVotes = await db.select()
        .from(paperVotes)
        .where(eq(paperVotes.userId, user.id));

      const relevanceScores = await db.select()
        .from(paperRelevanceScores)
        .where(eq(paperRelevanceScores.userId, user.id));

      const metrics = {
        totalVotes: userVotes.length,
        upvotes: userVotes.filter(v => v.vote === 1).length,
        downvotes: userVotes.filter(v => v.vote === -1).length,
        averageRelevanceScore: relevanceScores.length > 0 
          ? relevanceScores.reduce((acc, curr) => acc + curr.score, 0) / relevanceScores.length 
          : 0,
        averageConfidence: relevanceScores.length > 0
          ? relevanceScores.reduce((acc, curr) => acc + curr.confidence, 0) / relevanceScores.length
          : 0
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