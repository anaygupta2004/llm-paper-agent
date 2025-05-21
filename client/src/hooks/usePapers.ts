import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { Paper } from "@db/schema";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";

export function usePapers(preferences: string | null = null, page: number = 1, mode: 'annotation' | 'relevance' = 'annotation') {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

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
        const data = await response.json();
        
        // Handle API key requirement
        if (response.status === 400 && data.requiresApiKey) {
          toast({
            title: "API Key Required",
            description: data.message || "Please set your OpenAI API key in settings to enable paper search.",
            variant: "destructive"
          });
          // Trigger settings modal or navigation
          queryClient.setQueryData(['showSettingsModal'], true);
        }
        
        throw new Error(data.message || `${response.status}: ${response.statusText}`);
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