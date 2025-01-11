import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql } from "drizzle-orm";
import { analyzePaperRelevance } from "./openai";

interface RecommendationScore {
  paperId: number;
  score: number;
  confidence: number;
  needsFeedback: boolean;
  clusterScore?: number;
}

export async function getRecommendations(userId: string, preferences: string): Promise<RecommendationScore[]> {
  // Get user's voting history for personalization
  const userVotes = await db.select()
    .from(paperVotes)
    .where(eq(paperVotes.userId, userId));

  // Get papers that haven't been voted on
  const votedPaperIds = userVotes.map(v => v.paperId);
  const unvotedPapers = await db.select()
    .from(papers)
    .where(sql`${papers.id} NOT IN ${votedPaperIds}`);

  // Get previous relevance scores for this user
  const existingScores = await db.select()
    .from(paperRelevanceScores)
    .where(eq(paperRelevanceScores.userId, userId));

  // Get cluster information from similar users' votes
  const similarUsersVotes = await getSimilarUsersVotes(userId, preferences);

  // Calculate recommendations using active learning
  const recommendations: RecommendationScore[] = [];

  for (const paper of unvotedPapers) {
    // Check if we already have a relevance score
    const existingScore = existingScores.find(s => s.paperId === paper.id);

    if (existingScore) {
      // Use existing score but adjust confidence based on time and feedback consistency
      const ageInDays = (Date.now() - existingScore.createdAt.getTime()) / (1000 * 60 * 60 * 24);
      const adjustedConfidence = adjustConfidence(existingScore.confidence, {
        ageInDays,
        feedbackConsistency: calculateFeedbackConsistency(userId, paper.primaryCategory)
      });

      // Calculate cluster score based on similar users' preferences
      const clusterScore = calculateClusterScore(paper.id, similarUsersVotes);

      recommendations.push({
        paperId: paper.id,
        score: existingScore.score,
        confidence: adjustedConfidence,
        needsFeedback: adjustedConfidence < 50, // Request feedback for low confidence scores
        clusterScore
      });
    } else {
      // Get new relevance score from OpenAI
      try {
        const relevance = await analyzePaperRelevance(paper.abstract, preferences);

        // Store the score
        await db.insert(paperRelevanceScores).values({
          paperId: paper.id,
          userId,
          score: relevance.score,
          confidence: relevance.confidence,
          modelResponse: relevance
        });

        // Calculate cluster score for new papers
        const clusterScore = calculateClusterScore(paper.id, similarUsersVotes);

        recommendations.push({
          paperId: paper.id,
          score: relevance.score,
          confidence: relevance.confidence,
          needsFeedback: true, // Always request feedback for new scores
          clusterScore
        });
      } catch (error) {
        console.error(`Error analyzing paper ${paper.id}:`, error);
      }
    }
  }

  // Final ranking combining relevance scores and cluster preferences
  return recommendations.map(rec => ({
    ...rec,
    score: combineScores(rec.score, rec.clusterScore, rec.confidence)
  }));
}

async function getSimilarUsersVotes(userId: string, preferences: string) {
  // Find users with similar voting patterns
  const userVotes = await db.select()
    .from(paperVotes)
    .where(eq(paperVotes.userId, userId));

  const allUserVotes = await db.select()
    .from(paperVotes)
    .where(sql`${paperVotes.userId} != ${userId}`);

  // Group votes by user
  const votesByUser = allUserVotes.reduce((acc, vote) => {
    if (!acc[vote.userId]) acc[vote.userId] = [];
    acc[vote.userId].push(vote);
    return acc;
  }, {} as Record<string, typeof allUserVotes>);

  // Calculate similarity scores
  const similarityScores = Object.entries(votesByUser).map(([otherUserId, votes]) => {
    const similarity = calculateUserSimilarity(userVotes, votes);
    return { userId: otherUserId, similarity };
  });

  // Get top similar users' votes
  const topSimilarUsers = similarityScores
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 5);

  return allUserVotes.filter(vote => 
    topSimilarUsers.some(user => user.userId === vote.userId)
  );
}

function calculateUserSimilarity(userVotes: any[], otherVotes: any[]): number {
  // Find papers both users have voted on
  const commonPapers = userVotes.filter(v1 => 
    otherVotes.some(v2 => v2.paperId === v1.paperId)
  );

  if (commonPapers.length === 0) return 0;

  // Calculate similarity based on voting agreement
  const agreements = commonPapers.filter(v1 => 
    otherVotes.find(v2 => v2.paperId === v1.paperId && v2.vote === v1.vote)
  ).length;

  return agreements / commonPapers.length;
}

function calculateClusterScore(paperId: number, similarUsersVotes: any[]): number {
  const relevantVotes = similarUsersVotes.filter(v => v.paperId === paperId);
  if (!relevantVotes.length) return 0.5; // Neutral score if no similar users voted

  const positiveVotes = relevantVotes.filter(v => v.vote === 1).length;
  return positiveVotes / relevantVotes.length;
}

function adjustConfidence(
  baseConfidence: number, 
  factors: { ageInDays: number; feedbackConsistency: number }
): number {
  const timeDecay = Math.exp(-0.1 * factors.ageInDays);
  const consistencyBoost = factors.feedbackConsistency * 0.2;

  return Math.min(100, Math.max(0, 
    baseConfidence * timeDecay + consistencyBoost * 100
  ));
}

async function calculateFeedbackConsistency(userId: string, category: string): Promise<number> {
  // Get all votes from this user in the same category
  const relatedPapers = await db.select()
    .from(papers)
    .where(eq(papers.primaryCategory, category));

  const paperIds = relatedPapers.map(p => p.id);

  const votes = await db.select()
    .from(paperVotes)
    .where(and(
      eq(paperVotes.userId, userId),
      sql`${paperVotes.paperId} IN ${paperIds}`
    ));

  if (votes.length < 2) return 0.5; // Not enough data

  // Calculate consistency based on voting patterns
  const upvotes = votes.filter(v => v.vote === 1).length;
  const downvotes = votes.filter(v => v.vote === -1).length;

  // Higher score if votes are consistently positive or negative
  const consistency = Math.abs(upvotes - downvotes) / votes.length;
  return consistency;
}

function combineScores(
  relevanceScore: number,
  clusterScore: number | undefined,
  confidence: number
): number {
  const relevanceWeight = confidence / 100;
  const clusterWeight = 1 - relevanceWeight;

  return (
    relevanceScore * relevanceWeight +
    (clusterScore ?? 0.5) * 100 * clusterWeight
  );
}

export async function updateRecommendations(userId: string, paperId: number, vote: number) {
  // Get the paper's current scores
  const [currentScore] = await db.select()
    .from(paperRelevanceScores)
    .where(and(
      eq(paperRelevanceScores.userId, userId),
      eq(paperRelevanceScores.paperId, paperId)
    ));

  if (!currentScore) {
    return;
  }

  // Adjust scores based on user feedback
  const voteScore = vote === 1 ? 100 : 0;
  const learningRate = calculateAdaptiveLearningRate(userId, paperId);

  const newScore = Math.round(
    currentScore.score * (1 - learningRate) + voteScore * learningRate
  );

  const newConfidence = Math.min(100, currentScore.confidence + 10); // Increase confidence with feedback

  // Update the scores
  await db.update(paperRelevanceScores)
    .set({
      score: newScore,
      confidence: newConfidence,
      modelResponse: {
        ...currentScore.modelResponse,
        score: newScore,
        confidence: newConfidence,
        explanation: `Score adjusted based on user feedback: ${vote === 1 ? 'relevant' : 'not relevant'}`
      }
    })
    .where(eq(paperRelevanceScores.id, currentScore.id));

  // Update similar papers' scores
  await updateSimilarPapers(userId, paperId, vote);
}

async function calculateAdaptiveLearningRate(userId: string, paperId: number): Promise<number> {
  const [paper] = await db.select()
    .from(papers)
    .where(eq(papers.id, paperId));

  if (!paper) return 0.3; // Default learning rate

  // Get user's voting history in this category
  const categoryVotes = await db.select()
    .from(paperVotes)
    .where(and(
      eq(paperVotes.userId, userId),
      sql`${paperVotes.paperId} IN (
        SELECT id FROM papers WHERE primary_category = ${paper.primaryCategory}
      )`
    ));

  // Calculate consistency of votes in this category
  const consistency = calculateFeedbackConsistency(userId, paper.primaryCategory);

  // Adjust learning rate based on:
  // 1. Number of votes (more votes = lower learning rate)
  // 2. Consistency of votes (more consistent = higher learning rate)
  const baseRate = 0.3;
  const voteCount = categoryVotes.length;
  const experienceFactor = Math.exp(-0.1 * voteCount);
  const consistencyFactor = consistency * 0.5;

  return Math.min(0.5, Math.max(0.1,
    baseRate * experienceFactor + consistencyFactor
  ));
}

async function updateSimilarPapers(userId: string, paperId: number, vote: number) {
  // Get the voted paper
  const [votedPaper] = await db.select()
    .from(papers)
    .where(eq(papers.id, paperId));

  if (!votedPaper) {
    return;
  }

  // Get papers in the same category
  const similarPapers = await db.select()
    .from(papers)
    .where(eq(papers.primaryCategory, votedPaper.primaryCategory));

  // Get their scores
  const scores = await db.select()
    .from(paperRelevanceScores)
    .where(and(
      eq(paperRelevanceScores.userId, userId),
      sql`${paperRelevanceScores.paperId} IN ${similarPapers.map(p => p.id)}`
    ));

  // Update scores with a smaller learning rate
  const learningRate = 0.1;
  const voteScore = vote === 1 ? 100 : 0;

  for (const score of scores) {
    const newScore = Math.round(
      score.score * (1 - learningRate) + voteScore * learningRate
    );

    await db.update(paperRelevanceScores)
      .set({
        score: newScore,
        modelResponse: {
          ...score.modelResponse,
          score: newScore,
          explanation: `Score adjusted based on similar paper feedback`
        }
      })
      .where(eq(paperRelevanceScores.id, score.id));
  }
}