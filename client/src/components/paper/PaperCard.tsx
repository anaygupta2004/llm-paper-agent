import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ThumbsUp, ThumbsDown, ExternalLink, AlertCircle } from "lucide-react";
import { Paper } from "@db/schema";
import { useVotePaper } from "@/hooks/usePapers";
import { Progress } from "@/components/ui/progress";

interface PaperCardProps {
  paper: Paper & {
    relevanceScore?: number;
    confidence?: number;
  };
  showVoting?: boolean;
  mode?: "annotation" | "relevance";
}

export function PaperCard({ paper, showVoting = true, mode = "relevance" }: PaperCardProps) {
  const voteMutation = useVotePaper();

  const handleVote = (vote: 1 | -1) => {
    voteMutation.mutate({ paperId: paper.id, vote });
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-xl font-semibold">{paper.title}</h3>
            <p className="text-sm text-muted-foreground">{paper.authors}</p>
          </div>
          <a 
            href={paper.pdfUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-primary hover:opacity-80"
          >
            <ExternalLink className="h-5 w-5" />
          </a>
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm">{paper.abstract}</p>

        {mode === "relevance" && paper.relevanceScore !== undefined && (
          <div className="mt-4 space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Relevance Score</span>
              <span className="text-sm text-muted-foreground">{Math.round(paper.relevanceScore)}%</span>
            </div>
            <Progress value={paper.relevanceScore} className="h-2" />

            {paper.confidence && paper.confidence < 50 && (
              <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-500 mt-2">
                <AlertCircle className="h-4 w-4" />
                <span>Low confidence prediction - your feedback will help improve recommendations</span>
              </div>
            )}
          </div>
        )}
      </CardContent>

      {showVoting && (
        <CardFooter className="flex justify-between">
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleVote(1)}
              disabled={voteMutation.isPending}
            >
              <ThumbsUp className="h-4 w-4 mr-1" />
              Relevant
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleVote(-1)}
              disabled={voteMutation.isPending}
            >
              <ThumbsDown className="h-4 w-4 mr-1" />
              Not Relevant
            </Button>
          </div>
          <span className="text-sm text-muted-foreground">
            {new Date(paper.publishedDate).toLocaleDateString()}
          </span>
        </CardFooter>
      )}
    </Card>
  );
}