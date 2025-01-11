import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  firebaseId: text("firebase_id").unique().notNull(),
  email: text("email").unique().notNull(),
  preferences: jsonb("preferences").default({
    preferences: "",
    categories: [],
    openaiApiKey: null // Added API key storage in preferences
  }),
  createdAt: timestamp("created_at").defaultNow(),
});

export const papers = pgTable("papers", {
  id: serial("id").primaryKey(),
  arxivId: text("arxiv_id").unique().notNull(),
  title: text("title").notNull(),
  authors: text("authors").notNull(),
  abstract: text("abstract").notNull(),
  pdfUrl: text("pdf_url").notNull(),
  abstractUrl: text("abstract_url").notNull(),
  primaryCategory: text("primary_category").notNull(),
  publishedDate: timestamp("published_date").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const paperVotes = pgTable("paper_votes", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  paperId: integer("paper_id").references(() => papers.id),
  vote: integer("vote").notNull(), // 1 for upvote, -1 for downvote
  createdAt: timestamp("created_at").defaultNow(),
});

export const paperRelevanceScores = pgTable("paper_relevance_scores", {
  id: serial("id").primaryKey(),
  paperId: integer("paper_id").references(() => papers.id),
  userId: integer("user_id").references(() => users.id),
  score: integer("score").notNull(),
  confidence: integer("confidence").notNull(),
  modelResponse: jsonb("model_response").default({
    score: 0,
    confidence: 0,
    explanation: "",
    keywords: [],
    topicSimilarity: 0,
    methodologySimilarity: 0
  }),
  createdAt: timestamp("created_at").defaultNow(),
});

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
export type Paper = typeof papers.$inferSelect;
export type NewPaper = typeof papers.$inferInsert;
export type PaperVote = typeof paperVotes.$inferSelect;
export type NewPaperVote = typeof paperVotes.$inferInsert;
export type PaperRelevanceScore = typeof paperRelevanceScores.$inferSelect;
export type NewPaperRelevanceScore = typeof paperRelevanceScores.$inferInsert;

export interface UserPreferences {
  preferences: string;
  categories: string[];
  openaiApiKey?: string | null; // Added API key to preferences interface
}

export interface ModelResponse {
  score: number;
  confidence: number;
  explanation: string;
  keywords?: string[];
  topicSimilarity?: number;
  methodologySimilarity?: number;
}