import express from 'express';
import path from 'path';
import { log } from '../vite';
import { readdir, existsSync } from 'fs';
import { promisify } from 'util';

const readdirAsync = promisify(readdir);

export function setupStaticServing(app: express.Express) {
  const distPath = path.resolve(process.cwd(), "dist", "public");

  // Log static file configuration
  try {
    console.debug('[Static] Configuration:', {
      distPath,
      exists: existsSync(distPath),
      files: existsSync(distPath) ? readdirAsync(distPath) : []
    });
  } catch (error) {
    console.error('[Static] Error checking dist directory:', error);
  }

  // Serve static files with proper headers
  app.use(express.static(distPath, {
    maxAge: '1d',
    etag: true,
    lastModified: true,
    index: false,
    setHeaders: (res, filePath) => {
      // Set cache headers based on file type
      if (filePath.endsWith('.html')) {
        res.setHeader('Cache-Control', 'no-cache');
      } else if (filePath.match(/\.(js|css|png|jpg|jpeg|gif|ico|svg)$/)) {
        res.setHeader('Cache-Control', 'public, max-age=86400');
      }
    }
  }));

  // SPA fallback
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api')) {
      return next();
    }

    const indexPath = path.join(distPath, 'index.html');
    console.debug('[Static] Serving SPA fallback:', {
      path: req.path,
      file: indexPath
    });

    res.sendFile(indexPath, (err) => {
      if (err) {
        console.error('[Static] Error serving index.html:', err);
        res.status(500).send('Error loading application');
      }
    });
  });
}
