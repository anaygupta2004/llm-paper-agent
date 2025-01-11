import { VercelRequest, VercelResponse } from '@vercel/node';
import { db } from '../db';
import { corsMiddleware } from '../middleware/cors';
import { verifyAuthToken } from '../services/firebase';

// Health check endpoint
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Apply CORS
  await new Promise((resolve) => corsMiddleware(req, res, resolve));

  // Health check
  if (req.method === 'GET' && req.url === '/api/health') {
    return res.json({ status: 'ok', timestamp: new Date().toISOString() });
  }

  // Auth middleware for protected routes
  if (req.url?.startsWith('/api/') && req.url !== '/api/health') {
    try {
      const token = req.headers.authorization?.split("Bearer ")[1];
      if (!token) {
        return res.status(401).json({ error: "No authentication token provided" });
      }

      const decodedToken = await verifyAuthToken(token);
      req.user = decodedToken;
    } catch (error) {
      console.error("Auth error:", error);
      return res.status(401).json({ error: "Authentication failed" });
    }
  }

  // Route not found
  res.status(404).json({ error: 'Route not found' });
}
