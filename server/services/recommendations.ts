import { getDatabase, type Database } from "firebase-admin/database";
import { analyzePaperRelevance } from "./openai";

interface RecommendationScore {
  paperId: string;
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

export async function getRecommendations(userId: string, preferences: string): Promise<RecommendationScore[]> {
  try {
    const db = getDatabase();

    // Get user's voting history for personalization
    const votesRef = db.ref(`votes/${userId}`);
    const votesSnapshot = await votesRef.get();
    const userVotes = votesSnapshot.val() || {};

    // Get all papers
    const papersRef = db.ref('papers');
    const papersSnapshot = await papersRef.get();
    const papers = papersSnapshot.val() || {};

    // Get papers that haven't been voted on
    const unvotedPapers = Object.entries(papers)
      .filter(([paperId]) => !userVotes[paperId])
      .map(([_, paper]) => paper);

    // Get previous relevance scores for this user
    const scoresRef = db.ref(`paperRelevanceScores/${userId}`);
    const scoresSnapshot = await scoresRef.get();
    const existingScores = scoresSnapshot.val() || {};

    // Get category statistics
    const categoryStats = await calculateCategoryStatistics(userId);

    // Calculate recommendations
    const recommendations: RecommendationScore[] = [];

    for (const paper of unvotedPapers) {
      try {
        const existingScore = existingScores[paper.arxivId];

        if (existingScore) {
          const explorationScore = calculateExplorationScore({
            category: paper.primaryCategory,
            categoryStats,
            votingHistory: Object.keys(userVotes).length
          });

          const adjustedConfidence = adjustConfidence(existingScore.confidence, {
            ageInDays: existingScore.createdAt 
              ? (Date.now() - new Date(existingScore.createdAt).getTime()) / (1000 * 60 * 60 * 24)
              : 0,
            feedbackConsistency: await calculateFeedbackConsistency(userId, paper.primaryCategory),
            votingHistory: Object.keys(userVotes).length
          });

          const categoryScore = calculateCategoryScore(paper.primaryCategory, categoryStats);

          const finalScore = combineScores({
            relevanceScore: existingScore.score,
            explorationScore,
            categoryScore,
            confidence: adjustedConfidence,
            votingHistory: Object.keys(userVotes).length
          });

          recommendations.push({
            paperId: paper.arxivId,
            score: existingScore.score,
            confidence: adjustedConfidence,
            needsFeedback: shouldRequestFeedback(adjustedConfidence, Object.keys(userVotes).length),
            explorationScore,
            categoryScore,
            finalScore
          });
        } else {
          const relevance = await analyzePaperRelevance(
            paper.abstract,
            generateBroadPreferences(preferences, Object.keys(userVotes).length)
          );

          await db.ref(`paperRelevanceScores/${userId}/${paper.arxivId}`).set({
            score: relevance.score,
            confidence: relevance.confidence,
            modelResponse: relevance,
            createdAt: new Date().toISOString()
          });

          const explorationScore = calculateExplorationScore({
            category: paper.primaryCategory,
            categoryStats,
            votingHistory: Object.keys(userVotes).length
          });

          const categoryScore = calculateCategoryScore(paper.primaryCategory, categoryStats);

          const finalScore = combineScores({
            relevanceScore: relevance.score,
            explorationScore,
            categoryScore,
            confidence: relevance.confidence,
            votingHistory: Object.keys(userVotes).length
          });

          recommendations.push({
            paperId: paper.arxivId,
            score: relevance.score,
            confidence: relevance.confidence,
            needsFeedback: true,
            explorationScore,
            categoryScore,
            finalScore
          });
        }
      } catch (error) {
        console.error(`Error analyzing paper ${paper.arxivId}:`, error);
      }
    }

    return recommendations.sort((a, b) => (b.finalScore || 0) - (a.finalScore || 0));
  } catch (error) {
    console.error("Error getting recommendations:", error);
    return [];
  }
}

async function calculateCategoryStatistics(userId: string): Promise<CategoryStatistics> {
  try {
    const db = getDatabase();
    const votesRef = db.ref(`votes/${userId}`);
    const votesSnapshot = await votesRef.get();
    const userVotes = votesSnapshot.val() || {};

    const papersRef = db.ref('papers');
    const papersSnapshot = await papersRef.get();
    const papers = papersSnapshot.val() || {};

    const stats: CategoryStatistics = {};

    Object.entries(userVotes).forEach(([paperId, vote]: [string, any]) => {
      const paper = papers[paperId];
      if (!paper) return;

      if (!stats[paper.primaryCategory]) {
        stats[paper.primaryCategory] = {
          totalVotes: 0,
          positiveVotes: 0,
          uncertainty: 1
        };
      }

      stats[paper.primaryCategory].totalVotes++;
      if (vote.vote === 1) {
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
  } catch (error) {
    console.error("Error calculating category statistics:", error);
    return {};
  }
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

async function calculateFeedbackConsistency(userId: string, category: string): Promise<number> {
  try {
    const db = getDatabase();
    const papersRef = db.ref('papers');
    const papersSnapshot = await papersRef.get();
    const papers = papersSnapshot.val() || {};

    const votesRef = db.ref(`votes/${userId}`);
    const votesSnapshot = await votesRef.get();
    const votes = votesSnapshot.val() || {};

    const categoryVotes = Object.entries(votes)
      .filter(([paperId]) => papers[paperId]?.primaryCategory === category)
      .map(([_, vote]: [string, any]) => vote);

    if (categoryVotes.length < 2) return 0.5;

    const upvotes = categoryVotes.filter(v => v.vote === 1).length;
    const downvotes = categoryVotes.filter(v => v.vote === -1).length;

    return Math.abs(upvotes - downvotes) / categoryVotes.length;
  } catch (error) {
    console.error("Error calculating feedback consistency:", error);
    return 0.5;
  }
}

export async function updateRecommendations(userId: string, paperId: string, vote: number) {
  try {
    const db = getDatabase();
    const scoreRef = db.ref(`paperRelevanceScores/${userId}/${paperId}`);
    const scoreSnapshot = await scoreRef.get();
    const currentScore = scoreSnapshot.val();

    if (!currentScore) return;

    const categoryStats = await calculateCategoryStatistics(userId);
    const paperRef = db.ref(`papers/${paperId}`);
    const paperSnapshot = await paperRef.get();
    const paper = paperSnapshot.val();

    if (!paper) return;

    const learningRate = 0.3 * (categoryStats[paper.primaryCategory]?.uncertainty || 1);
    const voteScore = vote === 1 ? 100 : 0;
    const newScore = Math.round(currentScore.score * (1 - learningRate) + voteScore * learningRate);

    const categoryConfidence = categoryStats[paper.primaryCategory]?.totalVotes || 0;
    const confidenceBoost = Math.min(20, categoryConfidence * 2);
    const newConfidence = Math.min(100, currentScore.confidence + confidenceBoost);

    await db.ref(`paperRelevanceScores/${userId}/${paperId}`).set({
      score: newScore,
      confidence: newConfidence,
      modelResponse: {
        ...currentScore.modelResponse,
        score: newScore,
        confidence: newConfidence,
        explanation: `Score adjusted based on user feedback`
      },
      updatedAt: new Date().toISOString()
    });
  } catch (error) {
    console.error("Error updating recommendations:", error);
  }
}