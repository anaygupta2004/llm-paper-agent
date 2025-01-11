import { PaperCard } from "./PaperCard";
import { Paper } from "@db/schema";
import { Skeleton } from "@/components/ui/skeleton";

interface PaperListProps {
  papers: Paper[];
  loading: boolean;
  showVoting?: boolean;
}

export function PaperList({ papers, loading, showVoting }: PaperListProps) {
  if (loading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-[200px] w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {papers.map((paper) => (
        <PaperCard 
          key={paper.id} 
          paper={paper} 
          showVoting={showVoting} 
        />
      ))}
    </div>
  );
}
