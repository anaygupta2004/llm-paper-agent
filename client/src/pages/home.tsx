import { useState } from "react";
import { usePapers } from "@/hooks/usePapers";
import { PaperList } from "@/components/paper/PaperList";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

export default function Home() {
  const { user } = useAuth();
  const [preferences, setPreferences] = useState("");
  const [mode, setMode] = useState<"annotation" | "relevance">("annotation");
  const [page, setPage] = useState(1);
  
  const { data, isLoading } = usePapers(preferences, page);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
  };

  if (!user) {
    return (
      <Card className="p-6">
        <h1 className="text-2xl font-bold mb-4">Welcome to Paper Recommender</h1>
        <p>Please sign in to start discovering relevant research papers.</p>
      </Card>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card className="p-6">
        <h1 className="text-2xl font-bold mb-4">Research Paper Discovery</h1>
        
        <Tabs value={mode} onValueChange={(value) => setMode(value as "annotation" | "relevance")}>
          <TabsList className="mb-4">
            <TabsTrigger value="annotation">Annotation Mode</TabsTrigger>
            <TabsTrigger value="relevance">Relevance Mode</TabsTrigger>
          </TabsList>

          <TabsContent value="annotation">
            <p className="text-muted-foreground mb-4">
              Help improve our recommendations by annotating papers based on your interests.
            </p>
          </TabsContent>

          <TabsContent value="relevance">
            <p className="text-muted-foreground mb-4">
              View papers ranked by relevance to your research interests.
            </p>
          </TabsContent>
        </Tabs>

        <form onSubmit={handleSearch} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="preferences">Research Interests</Label>
            <Input
              id="preferences"
              placeholder="e.g., Deep Learning, Computer Vision, Natural Language Processing"
              value={preferences}
              onChange={(e) => setPreferences(e.target.value)}
            />
          </div>
          
          <Button type="submit" className="w-full">
            Find Papers
          </Button>
        </form>
      </Card>

      {data && (
        <div className="space-y-4">
          <PaperList
            papers={data.papers}
            loading={isLoading}
            showVoting={mode === "annotation"}
          />

          {data.totalPages > 1 && (
            <div className="flex justify-center gap-2">
              <Button
                variant="outline"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                onClick={() => setPage(p => Math.min(data.totalPages, p + 1))}
                disabled={page === data.totalPages}
              >
                Next
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
