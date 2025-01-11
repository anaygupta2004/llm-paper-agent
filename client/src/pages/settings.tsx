import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";

export default function Settings() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [preferences, setPreferences] = useState("");
  const [categories, setCategories] = useState("cs.LG,cs.AI,cs.CL");
  const [saving, setSaving] = useState(false);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      const response = await fetch("/api/settings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${await user?.getIdToken()}`
        },
        body: JSON.stringify({
          preferences,
          categories
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

      <Card>
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
    </div>
  );
}
