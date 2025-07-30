import { pgTable, text, serial, integer, boolean, jsonb, varchar, decimal } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const products = pgTable("products", {
  id: integer("id").primaryKey(),
  name: text("name").notNull(),
  category: text("category"),
  url: text("url"),
  originalText: text("original_text"),
  price: decimal("price", { precision: 10, scale: 2 }),
  imageUrl: text("image_url"),
  description: text("description"),
  brand: text("brand"),
  rating: decimal("rating", { precision: 3, scale: 2 }),
  reviewCount: integer("review_count").default(0),
});

export const searchFilters = pgTable("search_filters", {
  id: serial("id").primaryKey(),
  filterType: varchar("filter_type", { length: 50 }).notNull(),
  filterValue: text("filter_value").notNull(),
  displayName: text("display_name").notNull(),
  count: integer("count").default(0),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertProductSchema = createInsertSchema(products).pick({
  id: true,
  name: true,
  category: true,
  url: true,
  originalText: true,
  price: true,
  imageUrl: true,
  description: true,
  brand: true,
  rating: true,
  reviewCount: true,
});

export const productFilterSchema = z.object({
  category: z.string().optional(),
  minPrice: z.number().optional(),
  maxPrice: z.number().optional(),
  brand: z.string().optional(),
  search: z.string().optional(),
  page: z.number().default(1),
  limit: z.number().default(24),
  sortBy: z.enum(['relevance', 'price_low', 'price_high', 'newest', 'rating']).default('relevance'),
});

export const aiSearchSchema = z.object({
  query: z.string().min(1),
  filters: z.object({
    categories: z.array(z.string()).optional(),
    priceRange: z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    }).optional(),
    brands: z.array(z.string()).optional(),
  }).optional(),
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type Product = typeof products.$inferSelect;
export type InsertProduct = z.infer<typeof insertProductSchema>;
export type ProductFilter = z.infer<typeof productFilterSchema>;
export type AISearchRequest = z.infer<typeof aiSearchSchema>;
export type SearchFilter = typeof searchFilters.$inferSelect;
