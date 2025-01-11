import { useEffect, useState } from "react";
import { auth } from "@/lib/firebase";
import { useToast } from "@/hooks/use-toast";
import { User } from "firebase/auth";

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    // Listen for auth state changes
    const unsubscribe = auth.onAuthStateChanged(
      (user) => {
        console.debug("Auth state changed:", { 
          isAuthenticated: !!user,
          email: user?.email,
          emailVerified: user?.emailVerified,
          timestamp: new Date().toISOString()
        });

        setUser(user);
        setLoading(false);

        if (user) {
          // Show success toast only on sign in
          toast({
            title: "Successfully signed in",
            description: `Welcome${user.displayName ? ` ${user.displayName}` : ''}!`,
          });
        }
      },
      (error) => {
        console.error("Auth State Error:", error);
        toast({
          title: "Authentication Error",
          description: error.message || "There was a problem with authentication. Please try again.",
          variant: "destructive"
        });
        setLoading(false);
      }
    );

    // Log initial auth state for debugging
    console.debug("Initial auth state:", {
      currentUser: auth.currentUser?.email,
      isInitializing: loading,
      timestamp: new Date().toISOString()
    });

    return () => unsubscribe();
  }, [toast]);

  return { user, loading };
}