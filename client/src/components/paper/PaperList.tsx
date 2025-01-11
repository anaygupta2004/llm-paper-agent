import { PaperCard } from "./PaperCard";
import { Paper } from "@db/schema";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface PaperListProps {
  papers: (Paper & {
    relevanceScore?: number;
    confidence?: number;
  })[];
  loading: boolean;
  showVoting?: boolean;
  mode?: "annotation" | "relevance";
}

export function PaperList({ papers, loading, showVoting, mode = "relevance" }: PaperListProps) {
  if (loading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-[200px] w-full" />
        ))}
      </div>
    );
  }

  if (!papers.length) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {mode === "annotation" 
            ? "No papers need feedback at the moment. Check back later for more papers to annotate."
            : "No relevant papers found. Try adjusting your research interests or help improve recommendations by annotating some papers."}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      {papers.map((paper) => (
        <PaperCard 
          key={paper.id} 
          paper={paper} 
          showVoting={showVoting}
          mode={mode}
        />
      ))}
    </div>
  );
}