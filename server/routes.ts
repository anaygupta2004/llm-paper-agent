import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { verifyAuthToken } from "./services/firebase";
import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql } from "drizzle-orm";
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
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

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
      const allPapers = await db.select()
        .from(papers)
        .orderBy(desc(papers.publishedDate))
        .execute();

      // If searching or in relevance mode, analyze papers
      if (preferences || mode === "relevance") {
        console.log("\n============== ANALYZING PAPERS ==============");

        const scoredPapers = await Promise.all(
          allPapers.map(async (paper) => {
            try {
              console.log(`\nAnalyzing paper: ${paper.title}`);
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

        // Sort by relevance and filter out low-relevance papers
        const rankedPapers = scoredPapers
          .filter(paper => paper.relevanceScore >= 50)
          .sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

        console.log("\n============== TOP 20 PAPERS ==============");
        rankedPapers.slice(0, 20).forEach((paper, index) => {
          console.log(`
Paper #${index + 1}:
Title: ${paper.title}
Relevance Score: ${paper.relevanceScore}%
Confidence: ${paper.confidence}%
Explanation: ${paper.explanation}
Abstract: ${paper.abstract.substring(0, 300)}...
==============================================`);
        });

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
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  // Export annotations route
  app.get("/api/papers/export", requireAuth, async (req: Request, res: Response) => {
    try {
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      // Get user's votes with paper details
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

      // Format data for export
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

      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Disposition', 'attachment; filename=paper-annotations.json');
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
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

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

      // Get papers with relevance scores
      const votedPaperIds = votes.map(v => v.paperId);
      const relevanceScores = await db.select()
        .from(paperRelevanceScores)
        .where(eq(paperRelevanceScores.userId, user.id));

      // Handle empty voted papers array
      const votedPapers = votedPaperIds.length > 0 
        ? await db.select()
            .from(papers)
            .where(sql`${papers.id} IN (${votedPaperIds.join(",")})`)
        : [];

      // Combine papers with their relevance scores
      const papersWithScores = votedPapers.map(paper => ({
        ...paper,
        relevanceScore: relevanceScores.find(s => s.paperId === paper.id)?.score || 0,
      }));

      const metrics = {
        totalVotes: votes.length,
        upvotes: votes.filter(v => v.vote === 1).length,
        downvotes: votes.filter(v => v.vote === -1).length,
        averageRelevanceScore: relevanceScores.length 
          ? relevanceScores.reduce((acc, curr) => acc + curr.score, 0) / relevanceScores.length 
          : 0,
        averageConfidence: relevanceScores.length
          ? relevanceScores.reduce((acc, curr) => acc + curr.confidence, 0) / relevanceScores.length
          : 0,
        papers: papersWithScores,
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