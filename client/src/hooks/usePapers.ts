import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Paper } from "@db/schema";

export function usePapers(preferences: string, page: number = 1) {
  return useQuery({
    queryKey: ["/api/papers", preferences, page],
    queryFn: async () => {
      const response = await fetch(`/api/papers?preferences=${encodeURIComponent(preferences)}&page=${page}`);
      if (!response.ok) throw new Error("Failed to fetch papers");
      return response.json();
    }
  });
}

export function useVotePaper() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ paperId, vote }: { paperId: number, vote: 1 | -1 }) => {
      const response = await fetch("/api/papers/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paperId, vote })
      });
      if (!response.ok) throw new Error("Failed to vote");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/papers"] });
    }
  });
}

export function useMetrics() {
  return useQuery({
    queryKey: ["/api/metrics"],
    queryFn: async () => {
      const response = await fetch("/api/metrics");
      if (!response.ok) throw new Error("Failed to fetch metrics");
      return response.json();
    }
  });
}
