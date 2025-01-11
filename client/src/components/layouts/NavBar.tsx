import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/useAuth";
import { signInWithGoogle } from "@/lib/firebase";
import { Link } from "wouter";
import { Home, BarChart2, Settings, LogOut } from "lucide-react";
import { auth } from "@/lib/firebase";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

export function NavBar() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [isSigningIn, setIsSigningIn] = useState(false);

  const handleSignIn = async () => {
    if (isSigningIn) return;

    setIsSigningIn(true);
    try {
      console.debug("Starting Google sign-in process...");
      await signInWithGoogle();
      // Success toast will be shown by useAuth hook
    } catch (error: any) {
      console.error("Sign in error:", error);
      toast({
        title: "Authentication Error",
        description: error.message || "Failed to sign in. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSigningIn(false);
    }
  };

  const handleSignOut = async () => {
    try {
      await auth.signOut();
      toast({
        title: "Signed out successfully",
        description: "You have been signed out of your account.",
      });
    } catch (error: any) {
      console.error("Sign out error:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to sign out. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <nav className="border-b">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link href="/">
            <Button variant="ghost">
              <Home className="h-5 w-5 mr-2" />
              Home
            </Button>
          </Link>

          {user && (
            <>
              <Link href="/metrics">
                <Button variant="ghost">
                  <BarChart2 className="h-5 w-5 mr-2" />
                  Metrics
                </Button>
              </Link>
              <Link href="/settings">
                <Button variant="ghost">
                  <Settings className="h-5 w-5 mr-2" />
                  Settings
                </Button>
              </Link>
            </>
          )}
        </div>

        <div>
          {user ? (
            <div className="flex items-center gap-4">
              <span className="text-sm text-muted-foreground">
                {user.email}
              </span>
              <Button
                variant="outline"
                onClick={handleSignOut}
              >
                <LogOut className="h-5 w-5 mr-2" />
                Sign Out
              </Button>
            </div>
          ) : (
            <Button 
              onClick={handleSignIn}
              disabled={isSigningIn}
              className="min-w-[150px]"
            >
              {isSigningIn ? "Signing in..." : "Sign In with Google"}
            </Button>
          )}
        </div>
      </div>
    </nav>
  );
}