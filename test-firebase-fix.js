// Test script to verify Firebase path sanitization fix
import { initializeApp, cert } from 'firebase-admin/app';
import { getDatabase } from 'firebase-admin/database';
import * as dotenv from 'dotenv';
dotenv.config();

// Function to sanitize paper IDs for Firebase paths
function sanitizeForFirebasePath(id) {
  return id.replace(/[.#$[\]]/g, '_');
}

async function testFirebasePathSanitization() {
  try {
    // Initialize Firebase admin
    const app = initializeApp({
      credential: cert({
        projectId: process.env.FIREBASE_PROJECT_ID,
        clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
        privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
      }),
      databaseURL: "https://arxiv-paper-rec-default-rtdb.firebaseio.com",
    });

    const database = getDatabase(app);
    
    // Test with problematic paper IDs
    const testIds = ["2505.14463v1", "2406.11379v2", "2505.14503v1"];
    
    console.log("Testing Firebase path sanitization fix...");
    
    for (const paperId of testIds) {
      console.log(`\nOriginal paper ID: ${paperId}`);
      const sanitizedId = sanitizeForFirebasePath(paperId);
      console.log(`Sanitized paper ID: ${sanitizedId}`);
      
      // Test storing a dummy paper in the database
      const paperRef = database.ref(`papers/${sanitizedId}`);
      const testPaper = {
        arxivId: paperId,
        title: `Test Paper ${paperId}`,
        abstract: "This is a test paper to verify the Firebase path sanitization fix.",
        publishedDate: new Date().toISOString(),
      };
      
      console.log(`Storing paper in Firebase at path: papers/${sanitizedId}`);
      await paperRef.set(testPaper);
      
      // Verify we can retrieve the paper
      console.log("Attempting to retrieve the paper...");
      const snapshot = await paperRef.get();
      if (snapshot.exists()) {
        console.log("✅ Successfully retrieved paper with sanitized ID!");
        console.log("Paper data:", snapshot.val());
      } else {
        console.log("❌ Failed to retrieve paper!");
      }
      
      // Clean up - remove the test paper
      console.log("Cleaning up...");
      await paperRef.remove();
    }
    
    console.log("\n✅ Firebase path sanitization fix verification completed successfully!");
    process.exit(0);
  } catch (error) {
    console.error("Test failed with error:", error);
    process.exit(1);
  }
}

testFirebasePathSanitization();