import express, { type Request, Response, NextFunction } from "express";
import cors from "cors";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";

const app = express();

// Configure CORS based on environment
const isDevelopment = process.env.NODE_ENV !== 'production';
const corsOptions = {
  origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) {
      callback(null, true);
      return;
    }

    // Check if origin is allowed
    if (
      origin === 'https://arxiv-agent.replit.app' || // Production domain
      origin.endsWith('.replit.dev') || // Allow all Replit preview domains
      origin.includes('replit.com') ||   // Allow Replit editor domains
      origin.includes('google.com') ||   // Allow Google auth domains
      (isDevelopment && (
        origin.includes('localhost') ||
        origin.includes('127.0.0.1')
      ))
    ) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Request logging middleware with detailed error logging
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        const responseStr = JSON.stringify(capturedJsonResponse);
        logLine += ` :: ${responseStr.length > 50 ? responseStr.slice(0, 50) + '...' : responseStr}`;
      }
      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = registerRoutes(app);

  // Global error handler with detailed logging
  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    console.error('Server Error:', err);
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    // Log the full error stack for debugging
    console.error('Error stack:', err.stack);

    res.status(status).json({ error: message });
  });

  if (isDevelopment) {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const PORT = process.env.PORT || 5000;
  server.listen(PORT, "0.0.0.0", () => {
    log(`Server running on port ${PORT} in ${process.env.NODE_ENV || 'development'} mode`);
    // Log the current environment and configuration
    console.debug('Server configuration:', {
      env: app.get('env'),
      port: PORT,
      corsEnabled: true,
      nodeEnv: process.env.NODE_ENV,
      allowedOrigins: [
        'https://arxiv-agent.replit.app',
        '*.replit.dev',
        'replit.com',
        'accounts.google.com'
      ]
    });
  });
})();