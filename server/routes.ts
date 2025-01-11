import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { verifyAuthToken } from "./services/firebase";
import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql, inArray } from "drizzle-orm";
import { analyzePaperRelevance } from "./services/openai";
import { fetchAndStorePapers } from "./services/papers";
import type { UserPreferences, ModelResponse } from "@db/schema";

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
    const { preferences, page = "1", mode = "annotation" } = req.query;
    const limit = 10;
    const offset = (Number(page) - 1) * limit;

    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      // Fetch new papers if searching
      if (preferences) {
        await fetchAndStorePapers({
          categories: ["cs.LG", "cs.AI", "cs.CL"],
          maxResults: 100,
          dateRange: 7,
          preferences: preferences as string,
        });
      }

      // Get all papers
      const allPapers = await db.select().from(papers).orderBy(desc(papers.publishedDate));

      // If searching or in relevance mode, analyze papers
      if (preferences || mode === "relevance") {
        const scoredPapers = await Promise.all(
          allPapers.map(async (paper) => {
            try {
              const relevance = await analyzePaperRelevance(
                paper.abstract,
                preferences as string || (user.preferences as UserPreferences)?.preferences || ""
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

  // Export annotations route
  app.get("/api/papers/export", requireAuth, async (req: Request, res: Response) => {
    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      const votes = await db.select({
        paper: papers,
        vote: paperVotes,
        relevance: paperRelevanceScores
      })
      .from(paperVotes)
      .where(eq(paperVotes.userId, user.id))
      .innerJoin(papers, eq(papers.id, paperVotes.paperId))
      .leftJoin(paperRelevanceScores, and(
        eq(paperRelevanceScores.paperId, papers.id),
        eq(paperRelevanceScores.userId, user.id)
      ));

      const exportData = votes.map(({ paper, vote, relevance }) => ({
        title: paper.title,
        abstract: paper.abstract,
        url: paper.abstractUrl,
        userVote: vote.vote === 1 ? 'relevant' : 'not relevant',
        algorithmScore: relevance?.score || null,
        algorithmConfidence: relevance?.confidence || null,
        algorithmExplanation: (relevance?.modelResponse as ModelResponse)?.explanation || null,
        date: paper.publishedDate
      }));

      res.json(exportData);
    } catch (error) {
      console.error("Error exporting annotations:", error);
      res.status(500).json({ error: "Failed to export annotations" });
    }
  });

  // Vote route
  app.post("/api/papers/vote", requireAuth, async (req: Request, res: Response) => {
    const { paperId, vote } = req.body;

    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      // If vote is 0, remove the vote
      if (vote === 0) {
        await db.delete(paperVotes)
          .where(and(
            eq(paperVotes.userId, user.id),
            eq(paperVotes.paperId, paperId)
          ));
      } else {
        // Otherwise, upsert the vote
        await db.insert(paperVotes)
          .values({
            paperId,
            userId: user.id,
            vote: vote === 1 ? 1 : -1,
          })
          .onConflictDoUpdate({
            target: [paperVotes.userId, paperVotes.paperId],
            set: { vote: vote === 1 ? 1 : -1 }
          });
      }

      res.json({ success: true });
    } catch (error) {
      console.error("Error recording vote:", error);
      res.status(500).json({ error: "Failed to record vote" });
    }
  });

  // Updated metrics route with algorithm performance metrics
  app.get("/api/metrics", requireAuth, async (req: Request, res: Response) => {
    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      // Get all votes for this user
      const votes = await db.select().from(paperVotes).where(eq(paperVotes.userId, user.id));
      const votedPaperIds = votes.map(v => v.paperId);

      // Get papers with relevance scores
      const relevanceScores = await db.select()
        .from(paperRelevanceScores)
        .where(eq(paperRelevanceScores.userId, user.id));

      // Get voted papers with their categories
      const votedPapers = await db.select()
        .from(papers)
        .where(inArray(papers.id, votedPaperIds));

      // Calculate metrics by category
      const categoryMetrics: { [key: string]: {
        totalPapers: number;
        relevantPapers: number;
        correctPredictions: number;
        averageConfidence: number;
      }} = {};

      for (const paper of votedPapers) {
        const vote = votes.find(v => v.paperId === paper.id);
        const score = relevanceScores.find(s => s.paperId === paper.id);

        if (!vote || !score) continue;

        if (!categoryMetrics[paper.primaryCategory]) {
          categoryMetrics[paper.primaryCategory] = {
            totalPapers: 0,
            relevantPapers: 0,
            correctPredictions: 0,
            averageConfidence: 0,
          };
        }

        const metrics = categoryMetrics[paper.primaryCategory];
        metrics.totalPapers++;

        if (vote.vote === 1) {
          metrics.relevantPapers++;
        }

        const predicted = score.score >= 70;
        const actual = vote.vote === 1;
        if (predicted === actual) {
          metrics.correctPredictions++;
        }

        metrics.averageConfidence += score.confidence;
      }

      // Calculate overall metrics
      let totalPapers = 0;
      let totalRelevantPapers = 0;
      let totalCorrectPredictions = 0;

      Object.values(categoryMetrics).forEach(metrics => {
        totalPapers += metrics.totalPapers;
        totalRelevantPapers += metrics.relevantPapers;
        totalCorrectPredictions += metrics.correctPredictions;
      });

      // Calculate average relevance score and confidence
      const averageRelevanceScore = relevanceScores.length
        ? relevanceScores.reduce((acc, curr) => acc + curr.score, 0) / relevanceScores.length
        : 0;

      const averageConfidence = relevanceScores.length
        ? relevanceScores.reduce((acc, curr) => acc + curr.confidence, 0) / relevanceScores.length
        : 0;

      res.json({
        totalVotes: votes.length,
        upvotes: votes.filter(v => v.vote === 1).length,
        downvotes: votes.filter(v => v.vote === -1).length,
        totalPapers,
        totalRelevantPapers,
        totalCorrectPredictions,
        averageRelevanceScore,
        averageConfidence,
        categoryMetrics,
        learningProgress: Math.min(100, (votes.length / 20) * 100),
        precision: totalCorrectPredictions / (totalPapers || 1),
        recall: totalCorrectPredictions / (totalRelevantPapers || 1),
      });
    } catch (error) {
      console.error("Error fetching metrics:", error);
      res.status(500).json({ error: "Failed to fetch metrics" });
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

    // Check if user exists in our database
    const [existingUser] = await db.select()
      .from(users)
      .where(eq(users.firebaseId, decodedToken.uid));

    if (!existingUser) {
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