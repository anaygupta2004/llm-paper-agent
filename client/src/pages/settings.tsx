import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Eye, EyeOff, Key } from "lucide-react";

export default function Settings() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [preferences, setPreferences] = useState("");
  const [categories, setCategories] = useState("cs.LG,cs.AI,cs.CL");
  const [openaiApiKey, setOpenaiApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validatingKey, setValidatingKey] = useState(false);

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
          setOpenaiApiKey(data.openaiApiKey || "");
        }
      } catch (error) {
        console.error("Error loading settings:", error);
      }
    };

    if (user) {
      fetchSettings();
    }
  }, [user]);

  const validateApiKey = async (key: string) => {
    if (!key.startsWith('sk-')) {
      return { valid: false, message: 'Invalid API key format. Must start with "sk-"' };
    }
    return { valid: true };
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      if (openaiApiKey) {
        setValidatingKey(true);
        const validation = await validateApiKey(openaiApiKey);
        if (!validation.valid) {
          toast({
            title: "Invalid API Key",
            description: validation.message,
            variant: "destructive"
          });
          setValidatingKey(false);
          setSaving(false);
          return;
        }
        setValidatingKey(false);
      }

      const response = await fetch("/api/settings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${await user?.getIdToken()}`
        },
        body: JSON.stringify({
          preferences,
          categories: categories.split(",").map(c => c.trim()),
          openaiApiKey: openaiApiKey || null
        })
      });

      if (!response.ok) throw new Error("Failed to save settings");

      toast({
        title: "Settings saved",
        description: "Your preferences have been updated successfully."
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save settings. Please try again.",
        variant: "destructive"
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Research Preferences</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSave} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="preferences">Default Research Interests</Label>
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

            <Button type="submit" disabled={saving}>
              {saving ? "Saving..." : "Save Settings"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            OpenAI API Key
          </CardTitle>
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
                Provide your own OpenAI API key to use for paper analysis and recommendations. 
                Your key will be stored securely and used only for this application.
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                {openaiApiKey ? "✓ API key is set" : "No API key set"}
              </p>
            </div>

            <Button type="submit" disabled={saving || validatingKey}>
              {saving ? "Saving..." : validatingKey ? "Validating API Key..." : "Save API Key"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}