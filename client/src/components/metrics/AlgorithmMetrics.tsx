import { useMetrics } from "@/hooks/usePapers";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Brain, Target, Zap, TrendingUp } from "lucide-react";

export function AlgorithmMetrics() {
  const { data: metrics } = useMetrics();

  if (!metrics) return null;

  // Calculate precision: correct predictions / total predictions
  const precision = metrics.upvotes / (metrics.totalVotes || 1);
  
  // Calculate recall: correct predictions / total relevant papers
  const recall = metrics.upvotes / (metrics.totalRelevantPapers || 1);
  
  // F1 score: harmonic mean of precision and recall
  const f1Score = 2 * (precision * recall) / (precision + recall || 1);

  // Learning progress: how well the algorithm is learning from feedback
  const learningProgress = Math.min(100, (metrics.totalVotes / 20) * 100);

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Algorithm Precision
          </CardTitle>
          <Target className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">
                {(precision * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">
                Based on {metrics.totalVotes} votes
              </span>
            </div>
            <Progress value={precision * 100} className="h-2" />
            <p className="text-xs text-muted-foreground">
              How often the algorithm correctly identifies relevant papers
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Learning Progress
          </CardTitle>
          <Brain className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">
                {learningProgress.toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">
                {20 - metrics.totalVotes} more votes needed
              </span>
            </div>
            <Progress value={learningProgress} className="h-2" />
            <p className="text-xs text-muted-foreground">
              Algorithm training progress based on feedback received
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Prediction Quality
          </CardTitle>
          <Zap className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">
                {(f1Score * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">
                F1 Score
              </span>
            </div>
            <Progress value={f1Score * 100} className="h-2" />
            <p className="text-xs text-muted-foreground">
              Overall prediction quality combining precision and recall
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Average Relevance
          </CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">
                {metrics.averageRelevanceScore.toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">
                {metrics.averageConfidence.toFixed(1)}% confidence
              </span>
            </div>
            <Progress value={metrics.averageRelevanceScore} className="h-2" />
            <p className="text-xs text-muted-foreground">
              Average relevance score of recommended papers
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
