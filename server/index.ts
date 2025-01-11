import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { corsMiddleware } from "./middleware/cors";
import { setupStaticServing } from "./middleware/static";
import path from "path";

// Initialize express app
const app = express();

// Apply CORS first
app.use(corsMiddleware);

// Basic middlewares
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Request logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  console.debug('[Request]', {
    method: req.method,
    path: req.path,
    query: req.query,
    headers: {
      ...req.headers,
      cookie: undefined // Don't log cookies
    },
    timestamp: new Date().toISOString()
  });

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
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }
      log(logLine);
    }
  });

  next();
});

// Health check endpoint
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

(async () => {
  const server = registerRoutes(app);

  // Global error handler
  app.use((err: any, req: Request, res: Response, _next: NextFunction) => {
    console.error('[Server Error]', {
      error: err.message,
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

  // Setup appropriate server based on environment
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    setupStaticServing(app);
  }

  // Start server
  const PORT = Number(process.env.PORT) || 5000;
  server.listen(PORT, "0.0.0.0", () => {
    log(`Server running on port ${PORT} in ${process.env.NODE_ENV || 'development'} mode`);
    console.debug('[Server] Configuration:', {
      env: app.get('env'),
      port: PORT,
      nodeEnv: process.env.NODE_ENV,
      cors: true,
      static: app.get('env') !== 'development',
    });
  });
})();