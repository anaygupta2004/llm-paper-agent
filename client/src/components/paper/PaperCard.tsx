import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ThumbsUp, ThumbsDown, ExternalLink } from "lucide-react";
import { Paper } from "@db/schema";
import { useVotePaper } from "@/hooks/usePapers";

interface PaperCardProps {
  paper: Paper;
  showVoting?: boolean;
}

export function PaperCard({ paper, showVoting = true }: PaperCardProps) {
  const voteMutation = useVotePaper();

  const handleVote = (vote: 1 | -1) => {
    voteMutation.mutate({ paperId: paper.id, vote });
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start">
          <h3 className="text-xl font-semibold">{paper.title}</h3>
          <a 
            href={paper.pdfUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-primary hover:opacity-80"
          >
            <ExternalLink className="h-5 w-5" />
          </a>
        </div>
        <p className="text-sm text-muted-foreground">{paper.authors}</p>
      </CardHeader>
      
      <CardContent>
        <p className="text-sm">{paper.abstract}</p>
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
