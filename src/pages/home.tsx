import { Button } from "@/components/ui/button";
import { useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

interface ExportMutationResponse {
  success: boolean;
  message?: string;
}

export default function Home() {
  const exportMutation = useMutation<ExportMutationResponse, Error>({
    mutationFn: () => apiClient.post('/api/papers/export', {}),
  });

  return (
    <div>
      <Button
        onClick={() => exportMutation.mutate()}
        disabled={exportMutation.isPending}
      >
        Export Papers
      </Button>
    </div>
  );
}