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

      // Define allowed domains based on environment
      const allowedDomains = [
        // Production domains
        'arxiv-agent.vercel.app',       // Vercel production
        '.vercel.app',                  // All Vercel preview deployments
        // Development and auth domains
        'localhost',                    // Local development
        '127.0.0.1',                   // Local development
        '.replit.dev',                 // Replit preview domains
        '.replit.com',                 // Replit editor domains
        'accounts.google.com',         // Google auth
      ];

      const isAllowed = allowedDomains.some(domain => 
        hostname === domain || hostname.endsWith(domain)
      );

      if (isAllowed) {
        console.debug('[CORS] Allowed request:', {
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
        allowedDomains,
        isDevelopment,
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
    'Origin',
    'X-Auth-Token'
  ],
  exposedHeaders: ['X-Auth-Token'],
  maxAge: 86400, // 24 hours
  optionsSuccessStatus: 200
};

// Export CORS middleware
export const corsMiddleware = cors(corsOptions);