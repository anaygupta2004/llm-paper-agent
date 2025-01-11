import { initializeApp } from "firebase/app";
import { getAuth, signInWithRedirect, GoogleAuthProvider } from "firebase/auth";

// Fetch Firebase configuration from server
async function initializeFirebase() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      throw new Error("Failed to fetch Firebase configuration");
    }

    const config = await response.json();

    const firebaseConfig = {
      apiKey: config.firebaseApiKey,
      authDomain: `${config.firebaseProjectId}.firebaseapp.com`,
      projectId: config.firebaseProjectId,
      storageBucket: `${config.firebaseProjectId}.appspot.com`,
      appId: config.firebaseAppId,
    };

    return initializeApp(firebaseConfig);
  } catch (error) {
    console.error("Firebase initialization error:", error);
    throw error;
  }
}

// Initialize Firebase app
const app = await initializeFirebase();
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

// Configure additional scopes
googleProvider.addScope('https://www.googleapis.com/auth/userinfo.email');
googleProvider.addScope('https://www.googleapis.com/auth/userinfo.profile');

export async function signInWithGoogle() {
  try {
    await signInWithRedirect(auth, googleProvider);
  } catch (error: any) {
    console.error("Google Sign-In Error:", error);
    throw new Error(error.message || "Failed to sign in with Google");
  }
}

export async function signOut() {
  try {
    await auth.signOut();
  } catch (error: any) {
    console.error("Sign Out Error:", error);
    throw new Error(error.message || "Failed to sign out");
  }
}