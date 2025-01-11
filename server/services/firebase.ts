import { initializeApp, cert, type App } from "firebase-admin/app";
import { getAuth, type Auth } from "firebase-admin/auth";
import { type DecodedIdToken } from "firebase-admin/auth";

// Initialize Firebase Admin
const app: App = initializeApp({
  credential: cert({
    projectId: process.env.FIREBASE_PROJECT_ID,
    clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
    // Handle both formats of private key storage
    privateKey: (process.env.FIREBASE_PRIVATE_KEY || '').replace(/\\n/g, '\n'),
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
    const decodedToken = await auth.verifyIdToken(token);
    console.debug('[Firebase] Token verified for user:', {
      uid: decodedToken.uid,
      email: decodedToken.email,
      timestamp: new Date().toISOString()
    });
    return decodedToken;
  } catch (error: any) {
    console.error("[Firebase] Auth error:", {
      error: error.message,
      code: error.code,
      timestamp: new Date().toISOString()
    });
    throw error;
  }
}

/**
 * Utility function to extract and validate Firebase token from request
 */
export async function extractAndVerifyToken(authHeader?: string): Promise<DecodedIdToken> {
  if (!authHeader?.startsWith('Bearer ')) {
    throw new Error('No authentication token provided');
  }

  const token = authHeader.split('Bearer ')[1];
  return verifyAuthToken(token);
}