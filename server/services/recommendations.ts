import { db } from "@db";
import { papers, paperVotes, paperRelevanceScores, users } from "@db/schema";
import { eq, and, desc, sql } from "drizzle-orm";
import { analyzePaperRelevance } from "./openai";

interface RecommendationScore {
  paperId: number;
  score: number;
  confidence: number;
  needsFeedback: boolean;
}

export async function getRecommendations(userId: string, preferences: string): Promise<RecommendationScore[]> {
  // Get user's voting history
  const userVotes = await db.select()
    .from(paperVotes)
    .where(eq(paperVotes.userId, userId));
  
  // Get papers that haven't been voted on
  const votedPaperIds = userVotes.map(v => v.paperId);
  const unvotedPapers = await db.select()
    .from(papers)
    .where(sql`${papers.id} NOT IN ${votedPaperIds}`);

  // Get previous relevance scores
  const existingScores = await db.select()
    .from(paperRelevanceScores)
    .where(eq(paperRelevanceScores.userId, userId));

  // Calculate recommendations using active learning
  const recommendations: RecommendationScore[] = [];
  
  for (const paper of unvotedPapers) {
    // Check if we already have a relevance score
    const existingScore = existingScores.find(s => s.paperId === paper.id);
    
    if (existingScore) {
      // Use existing score but adjust confidence based on time
      const ageInDays = (Date.now() - existingScore.createdAt.getTime()) / (1000 * 60 * 60 * 24);
      const adjustedConfidence = Math.max(0, existingScore.confidence - (ageInDays * 2)); // Decay confidence over time
      
      recommendations.push({
        paperId: paper.id,
        score: existingScore.score,
        confidence: adjustedConfidence,
        needsFeedback: adjustedConfidence < 50 // Request feedback for low confidence scores
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

        recommendations.push({
          paperId: paper.id,
          score: relevance.score,
          confidence: relevance.confidence,
          needsFeedback: true // Always request feedback for new scores
        });
      } catch (error) {
        console.error(`Error analyzing paper ${paper.id}:`, error);
      }
    }
  }

  return recommendations;
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
  const learningRate = 0.3; // How quickly we adapt to feedback
  
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
