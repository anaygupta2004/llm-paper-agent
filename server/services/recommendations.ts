import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql, inArray } from "drizzle-orm";
import { analyzePaperRelevance, calculatePaperSimilarity } from "./openai";

interface RecommendationScore {
  paperId: number;
  score: number;
  confidence: number;
  needsFeedback: boolean;
  explorationScore?: number;
  categoryScore?: number;
  finalScore?: number;
}

interface CategoryStatistics {
  [category: string]: {
    totalVotes: number;
    positiveVotes: number;
    uncertainty: number;
  };
}

export async function getRecommendations(userId: number, preferences: string): Promise<RecommendationScore[]> {
  // Get user's voting history for personalization
  const userVotes = await db.query.paperVotes.findMany({
    where: eq(paperVotes.userId, userId)
  });

  // Get papers that haven't been voted on
  const votedPaperIds = userVotes.map(v => v.paperId);
  const unvotedPapers = await db.query.papers.findMany({
    where: votedPaperIds.length ? sql`${papers.id} NOT IN ${votedPaperIds}` : undefined
  });

  // Get previous relevance scores for this user
  const existingScores = await db.query.paperRelevanceScores.findMany({
    where: eq(paperRelevanceScores.userId, userId)
  });

  // Get category statistics
  const categoryStats = await calculateCategoryStatistics(userId);

  // Calculate recommendations
  const recommendations: RecommendationScore[] = [];

  for (const paper of unvotedPapers) {
    try {
      const existingScore = existingScores.find(s => s.paperId === paper.id);

      if (existingScore) {
        const explorationScore = calculateExplorationScore({
          category: paper.primaryCategory,
          categoryStats,
          votingHistory: userVotes.length
        });

        const adjustedConfidence = adjustConfidence(existingScore.confidence, {
          ageInDays: existingScore.createdAt 
            ? (Date.now() - existingScore.createdAt.getTime()) / (1000 * 60 * 60 * 24)
            : 0,
          feedbackConsistency: await calculateFeedbackConsistency(userId, paper.primaryCategory),
          votingHistory: userVotes.length
        });

        const categoryScore = calculateCategoryScore(paper.primaryCategory, categoryStats);

        const finalScore = combineScores({
          relevanceScore: existingScore.score,
          explorationScore,
          categoryScore,
          confidence: adjustedConfidence,
          votingHistory: userVotes.length
        });

        recommendations.push({
          paperId: paper.id,
          score: existingScore.score,
          confidence: adjustedConfidence,
          needsFeedback: shouldRequestFeedback(adjustedConfidence, userVotes.length),
          explorationScore,
          categoryScore,
          finalScore
        });
      } else {
        const relevance = await analyzePaperRelevance(
          paper.abstract,
          generateBroadPreferences(preferences, userVotes.length)
        );

        await db.insert(paperRelevanceScores).values({
          paperId: paper.id,
          userId,
          score: relevance.score,
          confidence: relevance.confidence,
          modelResponse: relevance
        });

        const explorationScore = calculateExplorationScore({
          category: paper.primaryCategory,
          categoryStats,
          votingHistory: userVotes.length
        });

        const categoryScore = calculateCategoryScore(paper.primaryCategory, categoryStats);

        const finalScore = combineScores({
          relevanceScore: relevance.score,
          explorationScore,
          categoryScore,
          confidence: relevance.confidence,
          votingHistory: userVotes.length
        });

        recommendations.push({
          paperId: paper.id,
          score: relevance.score,
          confidence: relevance.confidence,
          needsFeedback: true,
          explorationScore,
          categoryScore,
          finalScore
        });
      }
    } catch (error) {
      console.error(`Error analyzing paper ${paper.id}:`, error);
    }
  }

  return recommendations.sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0));
}

async function calculateCategoryStatistics(userId: number): Promise<CategoryStatistics> {
  const votes = await db.query.paperVotes.findMany({
    where: eq(paperVotes.userId, userId),
    with: {
      paper: true
    }
  });

  const stats: CategoryStatistics = {};

  votes.forEach(({ vote, paper }) => {
    if (!paper) return;

    if (!stats[paper.primaryCategory]) {
      stats[paper.primaryCategory] = {
        totalVotes: 0,
        positiveVotes: 0,
        uncertainty: 1
      };
    }

    stats[paper.primaryCategory].totalVotes++;
    if (vote === 1) {
      stats[paper.primaryCategory].positiveVotes++;
    }
  });

  Object.keys(stats).forEach(category => {
    const { totalVotes, positiveVotes } = stats[category];
    if (totalVotes > 0) {
      const alpha = positiveVotes + 1;
      const beta = totalVotes - positiveVotes + 1;
      stats[category].uncertainty = Math.sqrt(alpha * beta) / ((alpha + beta) * (alpha + beta + 1));
    }
  });

  return stats;
}

function calculateExplorationScore({
  category,
  categoryStats,
  votingHistory
}: {
  category: string;
  categoryStats: CategoryStatistics;
  votingHistory: number;
}): number {
  const categoryUncertainty = categoryStats[category]?.uncertainty || 1;
  const explorationDecay = Math.exp(-0.01 * votingHistory);
  return categoryUncertainty * explorationDecay * 100;
}

function calculateCategoryScore(category: string, stats: CategoryStatistics): number {
  const categoryStats = stats[category];
  if (!categoryStats || categoryStats.totalVotes === 0) return 50;
  return ((categoryStats.positiveVotes + 1) / (categoryStats.totalVotes + 2)) * 100;
}

function shouldRequestFeedback(confidence: number, votingHistory: number): boolean {
  const threshold = Math.min(80, 50 + (votingHistory / 10));
  return confidence < threshold;
}

function generateBroadPreferences(preferences: string, votingHistory: number): string {
  if (votingHistory < 10) {
    return `${preferences} Consider broader research areas and methodologies in these fields.`;
  } else if (votingHistory < 30) {
    return `${preferences} Focus on specific methodologies while maintaining awareness of related areas.`;
  }
  return preferences;
}

function combineScores({
  relevanceScore,
  explorationScore,
  categoryScore,
  confidence,
  votingHistory
}: {
  relevanceScore: number;
  explorationScore: number;
  categoryScore: number;
  confidence: number;
  votingHistory: number;
}): number {
  const explorationWeight = Math.max(0.1, 0.5 - (votingHistory / 100));
  const categoryWeight = Math.min(0.4, 0.1 + (votingHistory / 100));
  const relevanceWeight = 1 - (explorationWeight + categoryWeight);

  return (
    relevanceScore * relevanceWeight +
    explorationScore * explorationWeight +
    categoryScore * categoryWeight
  );
}

function adjustConfidence(
  baseConfidence: number,
  factors: {
    ageInDays: number;
    feedbackConsistency: number;
    votingHistory: number;
  }
): number {
  const timeDecay = Math.exp(-0.1 * factors.ageInDays);
  const consistencyBoost = factors.feedbackConsistency * 0.2;
  const experienceBoost = Math.min(0.3, factors.votingHistory / 100);

  return Math.min(100, Math.max(0,
    baseConfidence * timeDecay +
    consistencyBoost * 100 +
    experienceBoost * 100
  ));
}

async function calculateFeedbackConsistency(userId: number, category: string): Promise<number> {
  const relatedPapers = await db.query.papers.findMany({
    where: eq(papers.primaryCategory, category)
  });

  const paperIds = relatedPapers.map(p => p.id);

  const votes = await db.query.paperVotes.findMany({
    where: and(
      eq(paperVotes.userId, userId),
      inArray(paperVotes.paperId, paperIds)
    )
  });

  if (votes.length < 2) return 0.5;

  const upvotes = votes.filter(v => v.vote === 1).length;
  const downvotes = votes.filter(v => v.vote === -1).length;

  return Math.abs(upvotes - downvotes) / votes.length;
}

export async function updateRecommendations(userId: number, paperId: number, vote: number) {
  const [currentScore] = await db.select()
    .from(paperRelevanceScores)
    .where(and(
      eq(paperRelevanceScores.userId, userId),
      eq(paperRelevanceScores.paperId, paperId)
    ));

  if (!currentScore) return;

  const categoryStats = await calculateCategoryStatistics(userId);
  const paper = await db.query.papers.findFirst({
    where: eq(papers.id, paperId)
  });

  if (!paper) return;

  const learningRate = 0.3 * (categoryStats[paper.primaryCategory]?.uncertainty || 1);
  const voteScore = vote === 1 ? 100 : 0;
  const newScore = Math.round(currentScore.score * (1 - learningRate) + voteScore * learningRate);

  const categoryConfidence = categoryStats[paper.primaryCategory]?.totalVotes || 0;
  const confidenceBoost = Math.min(20, categoryConfidence * 2);
  const newConfidence = Math.min(100, currentScore.confidence + confidenceBoost);

  await db.update(paperRelevanceScores)
    .set({
      score: newScore,
      confidence: newConfidence,
      modelResponse: {
        ...currentScore.modelResponse,
        score: newScore,
        confidence: newConfidence,
        explanation: `Score adjusted based on user feedback`
      }
    })
    .where(eq(paperRelevanceScores.id, currentScore.id));
}
function calculateClusterScore(paperId: number, similarUsersVotes: any[]): number {
  const relevantVotes = similarUsersVotes.filter(v => v.paperId === paperId);
  if (!relevantVotes.length) return 0.5; // Neutral score if no similar users voted

  const positiveVotes = relevantVotes.filter(v => v.vote === 1).length;
  return positiveVotes / relevantVotes.length;
}

async function getSimilarUsersVotes(userId: string, preferences: string) {
  // Find users with similar voting patterns
  const userVotes = await db.select()
    .from(paperVotes)
    .where(eq(paperVotes.userId, userId));

  const allUserVotes = await db.select()
    .from(paperVotes)
    .where(sql`${paperVotes.userId} != ${userId}`);

  // Calculate similarity scores with all other users
  const similarityScores = await Promise.all(
    Array.from(new Set(allUserVotes.map(v => v.userId))).map(async (otherUserId) => {
      const similarity = await calculateUserSimilarity(
        userVotes,
        allUserVotes.filter(v => v.userId === otherUserId)
      );
      return { userId: otherUserId, similarity };
    })
  );

  // Get top similar users' votes
  const topSimilarUsers = similarityScores
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 5);

  return allUserVotes.filter(vote =>
    topSimilarUsers.some(user => user.userId === vote.userId)
  );
}

async function calculateUserSimilarity(userVotes: any[], otherVotes: any[]): Promise<number> {
  // Find papers both users have voted on
  const commonPapers = userVotes.filter(v1 =>
    otherVotes.some(v2 => v2.paperId === v1.paperId)
  );

  if (commonPapers.length === 0) return 0;

  // Get paper abstracts for content-based similarity
  const paperDetails = await Promise.all(
    commonPapers.map(async v => {
      const [paper] = await db.select()
        .from(papers)
        .where(eq(papers.id, v.paperId))
        .limit(1);
      return paper;
    })
  );

  // Calculate both vote agreement and content similarity
  const voteAgreement = commonPapers.filter(v1 =>
    otherVotes.find(v2 => v2.paperId === v1.paperId && v2.vote === v1.vote)
  ).length / commonPapers.length;

  let contentSimilarity = 0;
  for (let i = 0; i < paperDetails.length - 1; i++) {
    for (let j = i + 1; j < paperDetails.length; j++) {
      const similarity = await calculatePaperSimilarity(
        paperDetails[i].abstract,
        paperDetails[j].abstract
      );
      contentSimilarity += similarity;
    }
  }
  contentSimilarity /= (paperDetails.length * (paperDetails.length - 1) / 2) || 1;

  // Combine vote agreement and content similarity
  return (voteAgreement * 0.7 + (contentSimilarity / 100) * 0.3);
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