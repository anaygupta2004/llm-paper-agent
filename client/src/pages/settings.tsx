import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Eye, EyeOff, Key, AlertTriangle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function Settings() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [preferences, setPreferences] = useState("");
  const [categories, setCategories] = useState("cs.LG,cs.AI,cs.CL");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [hasApiKey, setHasApiKey] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validatingKey, setValidatingKey] = useState(false);
  const [isFirstLogin, setIsFirstLogin] = useState(false);

  useEffect(() => {
    // Load existing settings
    const fetchSettings = async () => {
      try {
        const response = await fetch("/api/settings", {
          headers: {
            Authorization: `Bearer ${await user?.getIdToken()}`
          }
        });

        if (response.ok) {
          const data = await response.json();
          setPreferences(data.preferences || "");
          setCategories(data.categories?.join(",") || "cs.LG,cs.AI,cs.CL");
          setHasApiKey(!!data.openaiApiKey);
          setIsFirstLogin(!data.openaiApiKey);
        }
      } catch (error) {
        console.error("Error loading settings:", error);
        toast({
          title: "Error",
          description: "Failed to load settings. Please try again.",
          variant: "destructive"
        });
      }
    };

    if (user) {
      fetchSettings();
    }
  }, [user]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      if (openaiApiKey) {
        setValidatingKey(true);
        const response = await fetch("/api/settings", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${await user?.getIdToken()}`
          },
          body: JSON.stringify({
            preferences,
            categories: categories.split(",").map(c => c.trim()),
            openaiApiKey
          })
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || "Failed to save settings");
        }

        setHasApiKey(true);
        setIsFirstLogin(false);

        toast({
          title: "Settings saved",
          description: "Your preferences and API key have been updated successfully."
        });
      } else if (!hasApiKey) {
        toast({
          title: "API Key Required",
          description: "Please provide your OpenAI API key to use the recommendation features.",
          variant: "destructive"
        });
        return;
      } else {
        // Save other settings without modifying API key
        await fetch("/api/settings", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${await user?.getIdToken()}`
          },
          body: JSON.stringify({
            preferences,
            categories: categories.split(",").map(c => c.trim())
          })
        });

        toast({
          title: "Settings saved",
          description: "Your preferences have been updated successfully."
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to save settings",
        variant: "destructive"
      });
    } finally {
      setSaving(false);
      setValidatingKey(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Settings</h1>

      {isFirstLogin && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Please provide your OpenAI API key to enable paper recommendations and analysis features.
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            OpenAI API Key
          </CardTitle>
          <CardDescription>
            Required for paper analysis and recommendations. Your key will be stored securely.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSave} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="openai-key">API Key</Label>
              <div className="relative">
                <Input
                  id="openai-key"
                  type={showApiKey ? "text" : "password"}
                  placeholder="sk-..."
                  value={openaiApiKey}
                  onChange={(e) => setOpenaiApiKey(e.target.value)}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showApiKey ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
              <p className="text-sm text-muted-foreground">
                Your OpenAI API key will be used exclusively for analyzing papers and generating recommendations.
              </p>
              <p className="text-sm font-medium">
                {hasApiKey ? (
                  <span className="text-green-600">✓ API key is set</span>
                ) : (
                  <span className="text-red-600">No API key set</span>
                )}
              </p>
            </div>

            <Button type="submit" disabled={saving || validatingKey}>
              {saving ? "Saving..." : validatingKey ? "Validating API Key..." : "Save API Key"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Research Preferences</CardTitle>
          <CardDescription>
            Configure your research interests and paper categories to receive better recommendations.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSave} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="preferences">Research Interests</Label>
              <Textarea
                id="preferences"
                placeholder="Describe your research interests..."
                value={preferences}
                onChange={(e) => setPreferences(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="categories">arXiv Categories</Label>
              <Input
                id="categories"
                placeholder="e.g., cs.LG,cs.AI,cs.CL"
                value={categories}
                onChange={(e) => setCategories(e.target.value)}
              />
              <p className="text-sm text-muted-foreground">
                Comma-separated list of arXiv categories to monitor
              </p>
            </div>

            <Button type="submit" disabled={saving || (!hasApiKey && !openaiApiKey)}>
              {saving ? "Saving..." : "Save Preferences"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}