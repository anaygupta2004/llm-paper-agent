import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { analyzePaperRelevance } from "./services/openai";
import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc } from "drizzle-orm";
import { verifyAuthToken } from "./services/firebase";

export function registerRoutes(app: Express): Server {
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

  // Paper routes
  app.get("/api/papers", requireAuth, async (req: Request, res: Response) => {
    const { preferences = "", page = "1" } = req.query;
    const limit = 10;
    const offset = (Number(page) - 1) * limit;

    try {
      const allPapers = await db.query.papers.findMany({
        orderBy: [desc(papers.publishedDate)],
        limit,
        offset
      });

      const scoredPapers = await Promise.all(
        allPapers.map(async (paper) => {
          const relevance = await analyzePaperRelevance(paper.abstract, preferences as string);

          await db.insert(paperRelevanceScores).values({
            paperId: paper.id,
            userId: req.user!.uid,
            score: relevance.score,
            confidence: relevance.confidence,
            modelResponse: relevance
          });

          return {
            ...paper,
            relevance
          };
        })
      );

      const totalCount = await db.select({ count: papers.id }).from(papers);
      const totalPages = Math.ceil(totalCount.length / limit);

      res.json({
        papers: scoredPapers,
        totalPages
      });
    } catch (error) {
      console.error("Error fetching papers:", error);
      res.status(500).json({ error: "Failed to fetch papers" });
    }
  });

  app.post("/api/papers/vote", requireAuth, async (req: Request, res: Response) => {
    const { paperId, vote } = req.body;

    try {
      await db.insert(paperVotes).values({
        paperId,
        userId: req.user!.uid,
        vote
      });

      res.json({ success: true });
    } catch (error) {
      console.error("Error recording vote:", error);
      res.status(500).json({ error: "Failed to record vote" });
    }
  });

  app.get("/api/metrics", requireAuth, async (req: Request, res: Response) => {
    try {
      const userVotes = await db.select()
        .from(paperVotes)
        .where(eq(paperVotes.userId, req.user!.uid));

      const relevanceScores = await db.select()
        .from(paperRelevanceScores)
        .where(eq(paperRelevanceScores.userId, req.user!.uid));

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

  const httpServer = createServer(app);
  return httpServer;
}