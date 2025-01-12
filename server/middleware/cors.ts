import cors from "cors";
import type { CorsOptions } from "cors";

const isDevelopment = process.env.NODE_ENV !== 'production';

export const corsOptions: CorsOptions = {
  origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) {
      callback(null, true);
      return;
    }

    try {
      const url = new URL(origin);
      const hostname = url.hostname;

      // In development, be more permissive
      if (isDevelopment) {
        callback(null, true);
        return;
      }

      // In production, only allow specific domains
      if (hostname.endsWith('.vercel.app') || 
          hostname === 'localhost' || 
          hostname === '127.0.0.1') {
        console.debug('[CORS] Allowed origin:', {
          origin,
          hostname,
          timestamp: new Date().toISOString()
        });
        callback(null, true);
        return;
      }

      console.warn('[CORS] Blocked request:', {
        origin,
        hostname,
        timestamp: new Date().toISOString()
      });

      callback(new Error('Not allowed by CORS'));
    } catch (error) {
      console.error('[CORS] Origin parsing error:', {
        origin,
        error,
        timestamp: new Date().toISOString()
      });
      callback(new Error('Invalid origin'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Requested-With',
    'Accept',
    'Origin'
  ],
  maxAge: 86400 // 24 hours
};

// Export CORS middleware
export const corsMiddleware = cors(corsOptions);