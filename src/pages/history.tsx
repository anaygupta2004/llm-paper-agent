import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { PaperList } from "@/components/paper/PaperList";
import { useEffect } from "react";
import type { Paper } from "@db/schema";

interface HistoryResponse {
  papers: (Paper & {
    relevanceScore?: number;
    confidence?: number;
    explanation?: string;
  })[];
  totalPages: number;
}

export default function History() {
  const { data, error, isLoading } = useQuery<HistoryResponse>({
    queryKey: ['/api/papers/history'],
    queryFn: () => apiClient.get('/api/papers/history')
  });

  useEffect(() => {
    console.debug("History page data:", data);
  }, [data]);

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Paper History</h1>
      <PaperList 
        papers={data?.papers || []} 
        mode="relevance"
      />
    </div>
  );
}