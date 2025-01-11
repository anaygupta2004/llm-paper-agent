import { Button } from "@/components/ui/button";
import { useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { useToast } from "@/hooks/use-toast";

interface ExportMutationResponse {
  success: boolean;
  message?: string;
  url?: string;
}

export default function Home() {
  const { toast } = useToast();

  const exportMutation = useMutation<ExportMutationResponse, Error>({
    mutationFn: () => apiClient.post('/api/papers/export', {}),
    onSuccess: (data) => {
      if (data.url) {
        window.open(data.url, '_blank');
      }
      toast({
        title: "Success",
        description: data.message || "Papers exported successfully",
      });
    },
    onError: (error) => {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    }
  });

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">ArXiv Research Assistant</h1>

      <Button
        onClick={() => exportMutation.mutate()}
        disabled={exportMutation.isPending}
      >
        {exportMutation.isPending ? "Exporting..." : "Export Papers"}
      </Button>
    </div>
  );
}