import { initializeApp } from "firebase/app";
import { getAuth, signInWithRedirect, GoogleAuthProvider } from "firebase/auth";

if (!import.meta.env.VITE_FIREBASE_API_KEY) {
  throw new Error("Firebase API key is missing");
}

if (!import.meta.env.VITE_FIREBASE_PROJECT_ID) {
  throw new Error("Firebase project ID is missing");
}

if (!import.meta.env.VITE_FIREBASE_APP_ID) {
  throw new Error("Firebase app ID is missing");
}

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: `${import.meta.env.VITE_FIREBASE_PROJECT_ID}.firebaseapp.com`,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: `${import.meta.env.VITE_FIREBASE_PROJECT_ID}.appspot.com`,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

export const signInWithGoogle = () => signInWithRedirect(auth, googleProvider);