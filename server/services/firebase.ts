import { initializeApp, cert, type App } from "firebase-admin/app";
import { getAuth, type Auth } from "firebase-admin/auth";
import { type DecodedIdToken } from "firebase-admin/auth";

// Initialize Firebase Admin
const app: App = initializeApp({
  credential: cert({
    projectId: process.env.FIREBASE_PROJECT_ID,
    clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
    privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
  }),
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

export async function verifyAuthToken(token: string): Promise<DecodedIdToken> {
  try {
    return await auth.verifyIdToken(token);
  } catch (error) {
    console.error("Error verifying auth token:", error);
    throw error;
  }
}