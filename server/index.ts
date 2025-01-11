import express, { type Request, Response, NextFunction } from "express";
import cors from "cors";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import path from "path";

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

    try {
      // Parse the origin URL to match subdomains
      const url = new URL(origin);
      const hostname = url.hostname;

      // Check if origin is allowed
      const allowedDomains = [
        'arxiv-agent.replit.app', // Production domain
        'replit.dev',            // Replit preview domains
        'replit.com',            // Replit editor domains
        'localhost',             // Local development
        '127.0.0.1'             // Local development
      ];

      const isAllowed = allowedDomains.some(domain => 
        hostname === domain || 
        hostname.endsWith(`.${domain}`)
      );

      if (isAllowed) {
        callback(null, true);
        return;
      }

      console.debug('CORS blocked request from:', {
        origin,
        hostname,
        allowedDomains,
        isDevelopment
      });

      callback(new Error('Not allowed by CORS'));
    } catch (error) {
      console.error('CORS origin parsing error:', error);
      callback(new Error('Invalid origin'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: [
    'Content-Type', 
    'Authorization', 
    'X-Requested-With',
    'Accept',
    'Origin'
  ],
  optionsSuccessStatus: 200
};

// Apply CORS first, before any other middleware
app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Request logging middleware with detailed error logging
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  // Detailed request logging
  console.debug('Request:', {
    method: req.method,
    path: req.path,
    headers: req.headers,
    query: req.query,
    timestamp: new Date().toISOString()
  });

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    const logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;

    if (res.statusCode >= 400) {
      console.error('Error Response:', {
        statusCode: res.statusCode,
        path: req.path,
        duration,
        response: capturedJsonResponse,
        headers: res.getHeaders()
      });
    } else if (path.startsWith("/api")) {
      log(logLine);
    }
  });

  next();
});

(async () => {
  const server = registerRoutes(app);

  // Global error handler with detailed logging
  app.use((err: any, req: Request, res: Response, _next: NextFunction) => {
    console.error('Server Error:', {
      error: err,
      stack: err.stack,
      path: req.path,
      method: req.method,
      timestamp: new Date().toISOString()
    });

    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ 
      error: message,
      path: req.path,
      timestamp: new Date().toISOString()
    });
  });

  if (isDevelopment) {
    await setupVite(app, server);
  } else {
    // In production, serve static files from the dist/public directory
    const distPath = path.resolve(process.cwd(), "dist", "public");

    // Log the static file configuration
    console.debug('Static file serving configuration:', {
      distPath,
      exists: require('fs').existsSync(distPath),
      files: require('fs').readdirSync(distPath)
    });

    // Serve static files with proper MIME types
    app.use(express.static(distPath, {
      maxAge: '1d',
      etag: true,
      index: false, // Don't serve index.html automatically
      setHeaders: (res, path) => {
        // Set proper cache headers
        if (path.endsWith('.html')) {
          res.setHeader('Cache-Control', 'no-cache');
        } else {
          res.setHeader('Cache-Control', 'public, max-age=86400');
        }
      }
    }));

    // SPA fallback - serve index.html for all non-API routes
    app.get('*', (req, res, next) => {
      if (req.path.startsWith('/api')) {
        return next();
      }

      const indexPath = path.join(distPath, 'index.html');
      console.debug('Serving SPA fallback:', {
        requestPath: req.path,
        servingFile: indexPath,
        exists: require('fs').existsSync(indexPath)
      });

      res.sendFile(indexPath, (err) => {
        if (err) {
          console.error('Error serving index.html:', err);
          res.status(500).send('Error serving application');
        }
      });
    });
  }

  const PORT = Number(process.env.PORT) || 5000;
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