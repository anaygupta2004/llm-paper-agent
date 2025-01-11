import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { verifyAuthToken } from "./services/firebase";
import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { and, eq, desc, sql, inArray } from "drizzle-orm";
import { analyzePaperRelevance } from "./services/openai";
import { fetchAndStorePapers } from "./services/papers";
import type { UserPreferences, ModelResponse } from "@db/schema";
import { validateApiKey, updateUserApiKey } from "./services/openai";

export function registerRoutes(app: Express): Server {
  // Settings endpoints
  app.get("/api/settings", requireAuth, async (req: Request, res: Response) => {
    try {
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      const userPreferences = (user.preferences as UserPreferences) || {
        preferences: "",
        categories: ["cs.LG", "cs.AI", "cs.CL"],
        openaiApiKey: null
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
      const [user] = await db.select()
        .from(users)
        .where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      const { preferences, categories, openaiApiKey } = req.body;

      // Get existing preferences
      const existingPreferences = (user.preferences as UserPreferences) || {
        preferences: "",
        categories: ["cs.LG", "cs.AI", "cs.CL"],
        openaiApiKey: null
      };

      // If OpenAI API key is provided, validate it
      if (openaiApiKey) {
        try {
          await validateApiKey(openaiApiKey);
          await updateUserApiKey(user.id, openaiApiKey);
        } catch (error) {
          console.error("API key validation error:", error);
          return res.status(400).json({
            error: error instanceof Error ? error.message : "Invalid OpenAI API key"
          });
        }
      }

      // Update user preferences
      await db.update(users)
        .set({
          preferences: {
            ...existingPreferences,
            preferences: preferences || existingPreferences.preferences,
            categories: categories || existingPreferences.categories,
            openaiApiKey: openaiApiKey || existingPreferences.openaiApiKey,
          }
        })
        .where(eq(users.id, user.id));

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
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      const userPreferences = user.preferences as UserPreferences;

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
      const allPapers = await db.select().from(papers).orderBy(desc(papers.publishedDate));

      // Get user's previous votes for confidence calculation
      const userVotes = await db.select()
        .from(paperVotes)
        .where(eq(paperVotes.userId, user.id));

      // If searching or in relevance mode, analyze papers
      if (preferences || mode === "relevance") {
        const scoredPapers = await Promise.all(
          allPapers.map(async (paper) => {
            try {
              const relevance = await analyzePaperRelevance(
                paper.abstract,
                preferences as string || userPreferences.preferences || "",
                user.id
              );

              // Calculate confidence based on voting history
              const voteCount = userVotes.length;
              const confidenceBoost = Math.min(30, voteCount * 0.5); // Max 30% boost from voting history
              const adjustedConfidence = Math.min(100, relevance.confidence + confidenceBoost);

              return {
                ...paper,
                relevanceScore: relevance.score,
                confidence: adjustedConfidence,
                explanation: relevance.explanation,
                needsFeedback: voteCount < 20 // Request more feedback during initial learning phase
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

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

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

  // Vote route with improved active learning
  app.post("/api/papers/vote", requireAuth, async (req: Request, res: Response) => {
    const { paperId, vote } = req.body;

    try {
      const [user] = await db.select().from(users).where(eq(users.firebaseId, req.user!.uid));

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      // Get the paper and its current relevance score
      const [paper] = await db.select()
        .from(papers)
        .where(eq(papers.id, paperId));

      if (!paper) {
        return res.status(404).json({ error: "Paper not found" });
      }

      // If vote is 0, remove the vote
      if (vote === 0) {
        await db.delete(paperVotes)
          .where(and(
            eq(paperVotes.userId, user.id),
            eq(paperVotes.paperId, paperId)
          ));
      } else {
        // Update or insert the vote
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

        // Update relevance scores with active learning
        const [currentScore] = await db.select()
          .from(paperRelevanceScores)
          .where(and(
            eq(paperRelevanceScores.userId, user.id),
            eq(paperRelevanceScores.paperId, paperId)
          ));

        if (currentScore) {
          // Calculate learning rate based on confidence
          const learningRate = Math.max(0.1, 0.5 - (currentScore.confidence / 200)); // Decreases as confidence increases
          const targetScore = vote === 1 ? 100 : 0;
          const newScore = Math.round(
            currentScore.score * (1 - learningRate) + targetScore * learningRate
          );

          // Update the score and increase confidence
          await db.update(paperRelevanceScores)
            .set({
              score: newScore,
              confidence: Math.min(100, currentScore.confidence + 5),
              modelResponse: {
                ...currentScore.modelResponse,
                score: newScore,
                learningProgress: Math.min(100, ((currentScore.modelResponse as ModelResponse).learningProgress || 0) + 5)
              }
            })
            .where(eq(paperRelevanceScores.id, currentScore.id));
        }
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

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      // Get all votes for this user
      const votes = await db.select()
        .from(paperVotes)
        .where(eq(paperVotes.userId, user.id));

      // Get papers with relevance scores
      const relevanceScores = await db.select()
        .from(paperRelevanceScores)
        .where(eq(paperRelevanceScores.userId, user.id));

      // Get papers for metric calculation
      const votedPaperIds = votes.map(v => v.paperId);
      const votedPapers = await db.select()
        .from(papers)
        .where(inArray(papers.id, votedPaperIds));

      // Calculate metrics by category
      const categoryMetrics: Record<string, {
        totalPapers: number;
        relevantPapers: number;
        correctPredictions: number;
        averageConfidence: number;
        truePositives: number;
        falsePositives: number;
        trueNegatives: number;
        falseNegatives: number;
      }> = {};

      // Initialize metrics
      for (const paper of votedPapers) {
        if (!categoryMetrics[paper.primaryCategory]) {
          categoryMetrics[paper.primaryCategory] = {
            totalPapers: 0,
            relevantPapers: 0,
            correctPredictions: 0,
            averageConfidence: 0,
            truePositives: 0,
            falsePositives: 0,
            trueNegatives: 0,
            falseNegatives: 0
          };
        }
      }

      // Calculate detailed metrics
      let totalCorrectPredictions = 0;
      let totalPapers = 0;

      for (const paper of votedPapers) {
        const vote = votes.find(v => v.paperId === paper.id);
        const score = relevanceScores.find(s => s.paperId === paper.id);

        if (!vote || !score) continue;

        const metrics = categoryMetrics[paper.primaryCategory];
        metrics.totalPapers++;
        totalPapers++;

        const predicted = score.score >= 70;
        const actual = vote.vote === 1;

        if (actual) {
          metrics.relevantPapers++;
        }

        if (predicted === actual) {
          metrics.correctPredictions++;
          totalCorrectPredictions++;
        }

        if (predicted && actual) metrics.truePositives++;
        if (predicted && !actual) metrics.falsePositives++;
        if (!predicted && !actual) metrics.trueNegatives++;
        if (!predicted && actual) metrics.falseNegatives++;

        metrics.averageConfidence += score.confidence;
      }

      // Calculate final metrics
      Object.values(categoryMetrics).forEach(metrics => {
        if (metrics.totalPapers > 0) {
          metrics.averageConfidence /= metrics.totalPapers;
        }
      });

      // Calculate overall metrics
      const overallMetrics = {
        accuracy: totalCorrectPredictions / totalPapers,
        totalVotes: votes.length,
        upvotes: votes.filter(v => v.vote === 1).length,
        downvotes: votes.filter(v => v.vote === -1).length,
        averageConfidence: relevanceScores.reduce((acc, curr) => acc + curr.confidence, 0) / relevanceScores.length,
        learningProgress: Math.min(100, (votes.length / 20) * 100)
      };

      res.json({
        overall: overallMetrics,
        categories: categoryMetrics,
        recentPerformance: calculateRecentPerformance(votes, relevanceScores),
        learningCurve: generateLearningCurve(votes, relevanceScores)
      });
    } catch (error) {
      console.error("Error fetching metrics:", error);
      res.status(500).json({ error: "Failed to fetch metrics" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

// Helper functions for metrics
function calculateRecentPerformance(votes: typeof paperVotes.$inferSelect[], scores: typeof paperRelevanceScores.$inferSelect[]) {
  const recentVotes = votes.slice(-20);
  let correctPredictions = 0;

  recentVotes.forEach(vote => {
    const score = scores.find(s => s.paperId === vote.paperId);
    if (score) {
      const predicted = score.score >= 70;
      const actual = vote.vote === 1;
      if (predicted === actual) correctPredictions++;
    }
  });

  return {
    recentAccuracy: recentVotes.length > 0 ? correctPredictions / recentVotes.length : 0,
    sampleSize: recentVotes.length
  };
}

function generateLearningCurve(votes: typeof paperVotes.$inferSelect[], scores: typeof paperRelevanceScores.$inferSelect[]) {
  const points = [];
  const windowSize = 10;

  for (let i = windowSize; i <= votes.length; i += windowSize) {
    const windowVotes = votes.slice(0, i);
    let correctPredictions = 0;

    windowVotes.forEach(vote => {
      const score = scores.find(s => s.paperId === vote.paperId);
      if (score) {
        const predicted = score.score >= 70;
        const actual = vote.vote === 1;
        if (predicted === actual) correctPredictions++;
      }
    });

    points.push({
      votes: i,
      accuracy: correctPredictions / windowVotes.length
    });
  }

  return points;
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
      // Create new user with default preferences
      await db.insert(users).values({
        firebaseId: decodedToken.uid,
        email: decodedToken.email!,
        preferences: {
          preferences: "",
          categories: ["cs.LG", "cs.AI", "cs.CL"],
          openaiApiKey: null
        }
      });
    }

    next();
  } catch (error) {
    console.error("Auth error:", error);
    res.status(401).json({ error: "Authentication failed" });
  }
};