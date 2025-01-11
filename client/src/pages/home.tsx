import { useState } from "react";
import { usePapers, useExportAnnotations } from "@/hooks/usePapers";
import { PaperList } from "@/components/paper/PaperList";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { motion, AnimatePresence } from "framer-motion";
import { Search, BookOpen, ThumbsUp, AlertCircle, Loader2, Download, Sparkles } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { Progress } from "@/components/ui/progress";

export default function Home() {
  const { user } = useAuth();
  const [searchInput, setSearchInput] = useState("");
  const [preferences, setPreferences] = useState<string | null>(null);
  const [mode, setMode] = useState<"annotation" | "relevance">("annotation");
  const [page, setPage] = useState(1);
  const [isSearching, setIsSearching] = useState(false);
  const [searchProgress, setSearchProgress] = useState(0);
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const exportMutation = useExportAnnotations();

  const { data, isLoading } = usePapers(preferences, page, mode);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (searchInput.trim().length < 3) {
      toast({
        title: "Invalid search",
        description: "Please enter at least 3 characters to search",
        variant: "destructive"
      });
      return;
    }

    setIsSearching(true);
    setSearchProgress(0);
    setPage(1);

    try {
      // Start progress animation
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress = Math.min(95, progress + 5);
        setSearchProgress(progress);
      }, 500);

      setPreferences(searchInput.trim());
      await queryClient.invalidateQueries({ queryKey: ['/api/papers'] });

      clearInterval(progressInterval);
      setSearchProgress(100);
      setTimeout(() => setSearchProgress(0), 500);
    } catch (error) {
      console.error('Search error:', error);
      toast({
        title: "Search failed",
        description: "Failed to search papers. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsSearching(false);
    }
  };

  const handleExport = async () => {
    try {
      const result = await exportMutation.mutateAsync();
      if (!result) return;

      // Create and download the file
      const url = window.URL.createObjectURL(new Blob([JSON.stringify(result)], { type: 'application/json' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = 'paper-annotations.json';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Export successful",
        description: "Your annotations have been exported successfully.",
      });
    } catch (error) {
      toast({
        title: "Export failed",
        description: "Failed to export annotations. Please try again.",
        variant: "destructive"
      });
    }
  };

  if (!user) {
    return (
      <div className="max-w-4xl mx-auto py-12 px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="p-8">
            <h1 className="text-4xl font-bold mb-4">Welcome to Paper Recommender</h1>
            <p className="text-lg text-muted-foreground mb-6">
              Discover relevant research papers with personalized recommendations powered by your feedback.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Feature
                icon={<Search className="h-8 w-8" />}
                title="Smart Discovery"
                description="Find papers tailored to your research interests using advanced AI"
              />
              <Feature
                icon={<BookOpen className="h-8 w-8" />}
                title="Active Learning"
                description="Our system learns from your feedback to improve your recommendations"
              />
              <Feature
                icon={<ThumbsUp className="h-8 w-8" />}
                title="Personal Relevance"
                description="Vote on papers to refine your personalized recommendations"
              />
            </div>
          </Card>
        </motion.div>
      </div>
    );
  }

  const showLoadingState = isLoading || isSearching;

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="p-6">
          <div className="flex justify-between items-start mb-6">
            <h1 className="text-3xl font-bold">Research Paper Discovery</h1>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExport}
              disabled={exportMutation.isLoading}
            >
              <Download className="h-4 w-4 mr-2" />
              Export Annotations
            </Button>
          </div>

          <Tabs value={mode} onValueChange={(value) => setMode(value as "annotation" | "relevance")} className="mb-6">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="annotation" className="flex items-center gap-2">
                <ThumbsUp className="h-4 w-4" />
                Annotation Mode
              </TabsTrigger>
              <TabsTrigger value="relevance" className="flex items-center gap-2">
                <Sparkles className="h-4 w-4" />
                AI Recommendations
              </TabsTrigger>
            </TabsList>

            <TabsContent value="annotation" className="mt-4">
              <div className="flex items-start gap-4 p-4 bg-muted/50 rounded-lg">
                <AlertCircle className="h-5 w-5 text-blue-500 mt-1" />
                <div>
                  <h3 className="font-medium mb-1">Annotation Mode</h3>
                  <p className="text-sm text-muted-foreground">
                    Help improve our recommendations by annotating papers. Your feedback trains the AI to better understand research relevance.
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="relevance" className="mt-4">
              <div className="flex items-start gap-4 p-4 bg-muted/50 rounded-lg">
                <Sparkles className="h-5 w-5 text-green-500 mt-1" />
                <div>
                  <h3 className="font-medium mb-1">AI Recommendations</h3>
                  <p className="text-sm text-muted-foreground">
                    Get personalized paper recommendations based on your interests and previous votes, powered by our advanced AI algorithm.
                  </p>
                </div>
              </div>
            </TabsContent>
          </Tabs>

          <form onSubmit={handleSearch} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="preferences">Research Interests</Label>
              <Textarea
                id="preferences"
                placeholder="Describe your research interests in detail. For example: 'I'm interested in deep learning applications in computer vision, particularly in medical image analysis using transformers.'"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                className="h-32"
                disabled={isSearching}
              />
              {searchInput.trim().length > 0 && searchInput.trim().length < 3 && (
                <p className="text-sm text-destructive">Please enter at least 3 characters</p>
              )}
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={isSearching || searchInput.trim().length < 3}
            >
              {isSearching ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Searching...
                </span>
              ) : (
                <>
                  <Search className="h-4 w-4 mr-2" />
                  Find Papers
                </>
              )}
            </Button>

            {searchProgress > 0 && (
              <div className="space-y-2">
                <Progress value={searchProgress} className="h-2" />
                <p className="text-sm text-muted-foreground text-center">
                  Analyzing papers for relevance...
                </p>
              </div>
            )}
          </form>
        </Card>
      </motion.div>

      <AnimatePresence mode="wait">
        {data && (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <div className="space-y-4">
              <PaperList
                papers={data.papers}
                loading={showLoadingState}
                mode={mode}
              />

              {data.totalPages > 1 && (
                <div className="flex justify-center gap-2 mt-8">
                  <Button
                    variant="outline"
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1 || showLoadingState}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setPage(p => Math.min(data.totalPages, p + 1))}
                    disabled={page === data.totalPages || showLoadingState}
                  >
                    Next
                  </Button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Feature({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="space-y-2">
      <div className="p-2 w-fit rounded-lg bg-primary/10">
        {icon}
      </div>
      <h3 className="font-semibold">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}