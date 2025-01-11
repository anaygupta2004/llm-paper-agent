import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { NavBar } from "@/components/layouts/NavBar";
import NotFound from "@/pages/not-found";
import Home from "@/pages/home";
import Metrics from "@/pages/metrics";
import Settings from "@/pages/settings";
import { useAuth } from "@/hooks/useAuth";
import { useEffect } from "react";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();

  useEffect(() => {
    // Debug logging for auth state
    console.debug("Protected route auth state:", { user, loading });
  }, [user, loading]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-lg">Loading authentication...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <div className="text-lg">Please sign in to access this page.</div>
      </div>
    );
  }

  return children;
}

function Router() {
  useEffect(() => {
    // Debug logging for initial route
    console.debug("Current route:", window.location.pathname);
    console.debug("Environment:", {
      isDev: import.meta.env.DEV,
      isProd: import.meta.env.PROD,
      baseUrl: import.meta.env.BASE_URL,
    });
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="container mx-auto px-4 py-8">
        <Switch>
          <Route path="/" component={Home} />
          <Route path="/metrics">
            <ProtectedRoute>
              <Metrics />
            </ProtectedRoute>
          </Route>
          <Route path="/settings">
            <ProtectedRoute>
              <Settings />
            </ProtectedRoute>
          </Route>
          <Route component={NotFound} />
        </Switch>
      </main>
    </div>
  );
}

function App() {
  useEffect(() => {
    // Debug logging for app initialization
    console.debug("App initialized");
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;