import { useEffect, useState } from "react";
import { auth } from "@/lib/firebase";
import { useToast } from "@/hooks/use-toast";
import { User, getRedirectResult } from "firebase/auth";

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    // Handle redirect result when the component mounts
    getRedirectResult(auth)
      .then((result) => {
        if (result) {
          toast({
            title: "Successfully signed in",
            description: `Welcome${result.user.displayName ? ` ${result.user.displayName}` : ''}!`,
          });
        }
      })
      .catch((error) => {
        console.error("Redirect Error:", error);
        if (error.code !== 'auth/redirect-cancelled-by-user') {
          let errorMessage = "An error occurred during sign in.";

          if (error.code === 'auth/configuration-not-found') {
            errorMessage = "Firebase configuration error. Please ensure the domain is authorized.";
          } else if (error.code === 'auth/popup-closed-by-user') {
            return; // Don't show error for user-initiated cancellation
          }

          toast({
            title: "Authentication Error",
            description: errorMessage,
            variant: "destructive"
          });
        }
      });

    // Listen for auth state changes
    const unsubscribe = auth.onAuthStateChanged(
      (user) => {
        setUser(user);
        setLoading(false);

        if (user) {
          console.log("User is signed in:", user.email);
        }
      },
      (error) => {
        console.error("Auth State Error:", error);
        toast({
          title: "Authentication Error",
          description: error.message,
          variant: "destructive"
        });
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, [toast]);

  return { user, loading };
}