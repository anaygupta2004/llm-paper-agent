import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ThumbsUp, ThumbsDown, ExternalLink, AlertCircle, ChevronDown, ChevronUp } from "lucide-react";
import { Paper } from "@db/schema";
import { useVotePaper } from "@/hooks/usePapers";
import { Progress } from "@/components/ui/progress";
import { useState } from "react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface PaperCardProps {
  paper: Paper & {
    relevanceScore?: number;
    confidence?: number;
    explanation?: string;
  };
  mode?: "annotation" | "relevance";
}

export function PaperCard({ paper, mode = "relevance" }: PaperCardProps) {
  const voteMutation = useVotePaper();
  const [isOpen, setIsOpen] = useState(false);

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

      <CardContent className="space-y-4">
        {mode === "relevance" && paper.relevanceScore !== undefined && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-lg font-semibold">Relevance Score</span>
              <span className="text-xl font-bold">{Math.round(paper.relevanceScore)}%</span>
            </div>
            <Progress value={paper.relevanceScore} className="h-2" />

            <Collapsible open={isOpen} onOpenChange={setIsOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full flex justify-between items-center p-2 hover:bg-accent">
                  <span className="font-medium">Analysis Details</span>
                  {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-2 space-y-2">
                {paper.confidence && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Confidence</span>
                    <span className="text-sm">{Math.round(paper.confidence)}%</span>
                  </div>
                )}

                {paper.explanation && (
                  <div className="mt-2 p-3 bg-muted rounded-lg">
                    <p className="text-sm whitespace-pre-wrap">{paper.explanation}</p>
                  </div>
                )}

                {paper.confidence && paper.confidence < 50 && (
                  <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-500 mt-2">
                    <AlertCircle className="h-4 w-4" />
                    <span>Low confidence prediction</span>
                  </div>
                )}
              </CollapsibleContent>
            </Collapsible>
          </div>
        )}

        <div className="text-sm">
          <p>{paper.abstract}</p>
        </div>
      </CardContent>

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
    </Card>
  );
}