import { initializeApp } from "firebase/app";
import { getAuth, signInWithPopup, GoogleAuthProvider } from "firebase/auth";
import { getDatabase, ref, set, get, query, orderByChild } from "firebase/database";

// Initialize Firebase configuration
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: `${import.meta.env.VITE_FIREBASE_PROJECT_ID}.firebaseapp.com`,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: `${import.meta.env.VITE_FIREBASE_PROJECT_ID}.appspot.com`,
  databaseURL: `https://${import.meta.env.VITE_FIREBASE_PROJECT_ID}-default-rtdb.firebaseio.com`,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getDatabase(app);
export const googleProvider = new GoogleAuthProvider();

// Configure additional scopes and parameters
googleProvider.addScope('https://www.googleapis.com/auth/userinfo.email');
googleProvider.addScope('https://www.googleapis.com/auth/userinfo.profile');
googleProvider.setCustomParameters({
  prompt: 'select_account'
});

// Firebase database helper functions
export async function savePaper(paperId: string, paperData: any) {
  try {
    await set(ref(db, `papers/${paperId}`), {
      ...paperData,
      createdAt: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error saving paper:", error);
    throw error;
  }
}

export async function getPaper(paperId: string) {
  try {
    const paperRef = ref(db, `papers/${paperId}`);
    const snapshot = await get(paperRef);
    return snapshot.exists() ? snapshot.val() : null;
  } catch (error) {
    console.error("Error getting paper:", error);
    throw error;
  }
}

export async function saveVote(userId: string, paperId: string, vote: number) {
  try {
    await set(ref(db, `votes/${userId}/${paperId}`), {
      vote,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error saving vote:", error);
    throw error;
  }
}

export async function getPaperVotes(paperId: string) {
  try {
    const votesRef = ref(db, 'votes');
    const votesQuery = query(votesRef, orderByChild(paperId));
    const snapshot = await get(votesQuery);
    return snapshot.exists() ? snapshot.val() : {};
  } catch (error) {
    console.error("Error getting votes:", error);
    throw error;
  }
}

export async function saveUserPreferences(userId: string, preferences: any) {
  try {
    await set(ref(db, `users/${userId}/preferences`), preferences);
  } catch (error) {
    console.error("Error saving preferences:", error);
    throw error;
  }
}

export async function signInWithGoogle() {
  try {
    // Check if Firebase is properly configured
    if (!import.meta.env.VITE_FIREBASE_API_KEY || 
        !import.meta.env.VITE_FIREBASE_PROJECT_ID || 
        !import.meta.env.VITE_FIREBASE_APP_ID) {
      throw new Error("Firebase configuration is incomplete. Please check environment variables.");
    }

    // Log authentication attempt
    console.debug('Attempting Google sign-in...');
    const result = await signInWithPopup(auth, googleProvider);
    console.debug('Sign-in successful:', result.user.email);

    // Save user data
    await set(ref(db, `users/${result.user.uid}`), {
      email: result.user.email,
      displayName: result.user.displayName,
      photoURL: result.user.photoURL,
      lastLogin: new Date().toISOString(),
    });

    return result.user;
  } catch (error: any) {
    console.error("Google Sign-In Error:", error);

    // Handle specific Firebase error codes
    switch (error.code) {
      case 'auth/operation-not-allowed':
        throw new Error("Google authentication is not enabled in Firebase Console. Please enable it in Authentication > Sign-in methods.");
      case 'auth/popup-closed-by-user':
        throw new Error("Sign-in cancelled");
      case 'auth/popup-blocked':
        throw new Error("Pop-up was blocked by the browser. Please allow pop-ups and try again.");
      case 'auth/unauthorized-domain':
        throw new Error(`This domain (${window.location.hostname}) is not authorized in Firebase Console. Please add it to authorized domains.`);
      case 'auth/configuration-not-found':
        throw new Error("Firebase configuration is invalid. Please check your Firebase Console settings.");
      case 'auth/internal-error':
        throw new Error("An internal error occurred. Please try again later.");
      default:
        throw new Error(error.message || "Failed to sign in with Google. Please try again.");
    }
  }
}

export async function signOut() {
  try {
    await auth.signOut();
  } catch (error: any) {
    console.error("Sign Out Error:", error);
    throw new Error("Failed to sign out. Please try again.");
  }
}

// Debug logging for Firebase configuration
const currentDomain = window.location.hostname;
console.debug("Firebase Config:", {
  hasApiKey: !!import.meta.env.VITE_FIREBASE_API_KEY,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  hasAppId: !!import.meta.env.VITE_FIREBASE_APP_ID,
  currentDomain,
  authDomain: `${import.meta.env.VITE_FIREBASE_PROJECT_ID}.firebaseapp.com`,
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD
});