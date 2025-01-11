import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { Paper } from "@db/schema";
import { useAuth } from "@/hooks/useAuth";

export function usePapers(preferences: string | null = null, page: number = 1, mode: 'annotation' | 'relevance' = 'annotation') {
  const { user } = useAuth();

  return useQuery({
    queryKey: ['/api/papers', preferences, page, mode],
    queryFn: async () => {
      if (!user) return { papers: [], totalPages: 0 };

      const params = new URLSearchParams({
        page: page.toString(),
        mode
      });

      if (preferences) {
        params.set('preferences', preferences);
      }

      const response = await fetch(`/api/papers?${params}`, {
        headers: {
          Authorization: `Bearer ${await user.getIdToken()}`
        }
      });

      if (!response.ok) {
        if (response.status >= 500) {
          throw new Error(`${response.status}: ${response.statusText}`);
        }

        throw new Error(`${response.status}: ${await response.text()}`);
      }

      return response.json();
    },
    enabled: !!user,
    staleTime: 1000 * 60 * 5, // 5 minutes
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

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to vote');
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/papers'] });
    }
  });
}

export function useExportAnnotations() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      if (!user) return null;

      const response = await fetch('/api/papers/export', {
        headers: {
          Authorization: `Bearer ${await user.getIdToken()}`
        }
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to export annotations');
      }

      return response.blob();
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

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || 'Failed to fetch metrics');
      }

      return response.json();
    },
    enabled: !!user
  });
}