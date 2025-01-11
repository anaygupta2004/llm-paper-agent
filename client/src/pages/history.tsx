import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { PaperList } from "@/components/paper/PaperList";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

type SortOption = "date" | "relevance" | "vote";

export default function History() {
  const { user } = useAuth();
  const [sortBy, setSortBy] = useState<SortOption>("date");

  const { data, isLoading } = useQuery({
    queryKey: ['/api/papers/history', { sortBy }],
    enabled: !!user,
  });

  if (!user) {
    return (
      <div className="max-w-4xl mx-auto py-12 px-4">
        <Card>
          <CardContent className="py-8">
            <p className="text-center text-muted-foreground">
              Please sign in to view your paper history.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Paper History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-6">
            <Label htmlFor="sort">Sort by</Label>
            <Select value={sortBy} onValueChange={(value) => setSortBy(value as SortOption)}>
              <SelectTrigger id="sort" className="w-[180px]">
                <SelectValue placeholder="Sort by..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="date">Date</SelectItem>
                <SelectItem value="relevance">Relevance Score</SelectItem>
                <SelectItem value="vote">Your Vote</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {isLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <PaperList 
              papers={data?.papers || []} 
              loading={isLoading}
              mode="history"
              showVoting={false}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
