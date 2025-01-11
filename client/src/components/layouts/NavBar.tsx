import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/useAuth";
import { signInWithGoogle } from "@/lib/firebase";
import { Link } from "wouter";
import { Home, BarChart2, Settings, LogOut } from "lucide-react";
import { auth } from "@/lib/firebase";

export function NavBar() {
  const { user } = useAuth();

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
                onClick={() => auth.signOut()}
              >
                <LogOut className="h-5 w-5 mr-2" />
                Sign Out
              </Button>
            </div>
          ) : (
            <Button onClick={signInWithGoogle}>
              Sign In with Google
            </Button>
          )}
        </div>
      </div>
    </nav>
  );
}
