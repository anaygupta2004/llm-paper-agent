import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Paper } from "@db/schema";
import { useAuth } from "@/hooks/useAuth";

export function usePapers(preferences: string, page: number = 1) {
  const { user } = useAuth();

  return useQuery({
    queryKey: ['/api/papers', preferences, page],
    queryFn: async () => {
      if (!user) return { papers: [], totalPages: 0 };

      const response = await fetch(`/api/papers?preferences=${encodeURIComponent(preferences)}&page=${page}`, {
        headers: {
          Authorization: `Bearer ${await user.getIdToken()}`
        }
      });
      if (!response.ok) throw new Error('Failed to fetch papers');
      return response.json();
    },
    enabled: !!user
  });
}

export function useVotePaper() {
  const queryClient = useQueryClient();
  const { user } = useAuth();

  return useMutation({
    mutationFn: async ({ paperId, vote }: { paperId: number, vote: 1 | -1 }) => {
      if (!user) throw new Error('Must be logged in to vote');

      const response = await fetch("/api/papers/vote", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          Authorization: `Bearer ${await user.getIdToken()}`
        },
        body: JSON.stringify({ paperId, vote })
      });
      if (!response.ok) throw new Error('Failed to vote');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/papers'] });
    }
  });
}

export function useMetrics() {
  const { user } = useAuth();

  return useQuery({
    queryKey: ['/api/metrics'],
    queryFn: async () => {
      if (!user) return null;

      const response = await fetch('/api/metrics', {
        headers: {
          Authorization: `Bearer ${await user.getIdToken()}`
        }
      });
      if (!response.ok) throw new Error('Failed to fetch metrics');
      return response.json();
    },
    enabled: !!user
  });
}