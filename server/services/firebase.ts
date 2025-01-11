import { initializeApp, cert, type App } from "firebase-admin/app";
import { getAuth, type Auth } from "firebase-admin/auth";
import { type DecodedIdToken } from "firebase-admin/auth";

// Initialize Firebase Admin with service account
const app: App = initializeApp({
  projectId: process.env.FIREBASE_PROJECT_ID,
});

export const auth: Auth = getAuth(app);

// Define types for the extended Express Request
declare global {
  namespace Express {
    interface Request {
      user?: DecodedIdToken;
    }
  }
}

// Middleware to verify Firebase ID tokens
export async function verifyAuthToken(token: string): Promise<DecodedIdToken> {
  try {
    return await auth.verifyIdToken(token);
  } catch (error) {
    console.error("Error verifying auth token:", error);
    throw new Error("Invalid authentication token");
  }
}