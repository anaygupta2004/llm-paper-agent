import { useState } from "react";
import { usePapers } from "@/hooks/usePapers";
import { PaperList } from "@/components/paper/PaperList";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { motion, AnimatePresence } from "framer-motion";
import { Search, BookOpen, ThumbsUp, AlertCircle, Loader2 } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";

export default function Home() {
  const { user } = useAuth();
  const [searchInput, setSearchInput] = useState("");
  const [preferences, setPreferences] = useState("");
  const [mode, setMode] = useState<"annotation" | "relevance">("annotation");
  const [page, setPage] = useState(1);
  const [isSearching, setIsSearching] = useState(false);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { data, isLoading, refetch } = usePapers(preferences, page, mode);

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
    setPage(1);

    try {
      // Update preferences to trigger the search
      setPreferences(searchInput.trim());
      // Invalidate existing queries to force a fresh fetch
      await queryClient.invalidateQueries({ queryKey: ['/api/papers'] });
      await refetch();
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
          <h1 className="text-3xl font-bold mb-6">Research Paper Discovery</h1>

          <Tabs value={mode} onValueChange={(value) => setMode(value as "annotation" | "relevance")} className="mb-6">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="annotation" className="flex items-center gap-2">
                <ThumbsUp className="h-4 w-4" />
                Annotation Mode
              </TabsTrigger>
              <TabsTrigger value="relevance" className="flex items-center gap-2">
                <Search className="h-4 w-4" />
                Relevance Mode
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
                <Search className="h-5 w-5 text-green-500 mt-1" />
                <div>
                  <h3 className="font-medium mb-1">Relevance Mode</h3>
                  <p className="text-sm text-muted-foreground">
                    View papers ranked by relevance to your research interests, powered by our advanced recommendation engine.
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
                loading={isSearching}
                showVoting={mode === "annotation"}
                mode={mode}
              />

              {data.totalPages > 1 && (
                <div className="flex justify-center gap-2 mt-8">
                  <Button
                    variant="outline"
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1 || isSearching}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setPage(p => Math.min(data.totalPages, p + 1))}
                    disabled={page === data.totalPages || isSearching}
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