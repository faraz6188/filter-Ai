import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { ProductService } from "./services/productService";
import { analyzeSearchIntent, generateProductFilters } from "./services/gemini";
import { productFilterSchema, aiSearchSchema } from "@shared/schema";
import { z } from "zod";

const productService = new ProductService();

export async function registerRoutes(app: Express): Promise<Server> {
  // Get products with filtering, searching, and pagination
  app.get("/api/products", async (req, res) => {
    try {
      const filters = productFilterSchema.parse({
        category: req.query.category as string,
        minPrice: req.query.minPrice ? Number(req.query.minPrice) : undefined,
        maxPrice: req.query.maxPrice ? Number(req.query.maxPrice) : undefined,
        brand: req.query.brand as string,
        search: req.query.search as string,
        page: req.query.page ? Number(req.query.page) : 1,
        limit: req.query.limit ? Number(req.query.limit) : 24,
        sortBy: req.query.sortBy as any || 'relevance',
      });

      const result = productService.getProducts(filters);
      res.json(result);
    } catch (error) {
      console.error("Error fetching products:", error);
      res.status(500).json({ error: "Failed to fetch products" });
    }
  });

  // Get filter options (categories, brands, price ranges)
  app.get("/api/filters", async (req, res) => {
    try {
      const categories = productService.getCategories();
      const brands = productService.getBrands();
      const priceRanges = productService.getPriceRanges();

      res.json({
        categories: categories.slice(0, 20), // Limit to top 20
        brands: brands.slice(0, 10), // Limit to top 10
        priceRanges,
      });
    } catch (error) {
      console.error("Error fetching filters:", error);
      res.status(500).json({ error: "Failed to fetch filters" });
    }
  });

  // Get AI-powered smart filters
  app.get("/api/filters/smart", async (req, res) => {
    try {
      const query = req.query.q as string || '';
      const smartFilters = await productService.getAISmartFilters(query);
      res.json(smartFilters);
    } catch (error) {
      console.error("Error fetching smart filters:", error);
      res.status(500).json({ error: "Failed to fetch smart filters" });
    }
  });

  // AI-powered search endpoint
  app.post("/api/search/ai", async (req, res) => {
    try {
      const { query, filters } = aiSearchSchema.parse(req.body);

      // Use Gemini AI to analyze search intent
      const searchIntent = await analyzeSearchIntent(query);
      
      // Generate smart filters based on the search query
      const aiFilters = await generateProductFilters(query, {
        categories: productService.getCategories().map(c => c.name),
        brands: productService.getBrands().map(b => b.name),
      });

      // Combine AI-generated filters with user-provided filters
      const combinedFilters = {
        search: query,
        category: aiFilters.category || filters?.categories?.[0],
        minPrice: aiFilters.priceRange?.min || filters?.priceRange?.min,
        maxPrice: aiFilters.priceRange?.max || filters?.priceRange?.max,
        brand: aiFilters.brand || filters?.brands?.[0],
        page: 1,
        limit: 24,
        sortBy: aiFilters.sortBy || 'relevance',
      };

      let result = productService.getProducts(combinedFilters);

      // Fallback: if no products, try just price, then just category, then just search
      if (result.products.length === 0) {
        // Try just price
        if (combinedFilters.minPrice !== undefined || combinedFilters.maxPrice !== undefined) {
          const priceOnly = { ...combinedFilters, category: undefined, brand: undefined };
          result = productService.getProducts(priceOnly);
        }
      }
      if (result.products.length === 0) {
        // Try just category
        if (combinedFilters.category) {
          const catOnly = { ...combinedFilters, minPrice: undefined, maxPrice: undefined, brand: undefined };
          result = productService.getProducts(catOnly);
        }
      }
      if (result.products.length === 0) {
        // Try just search
        const searchOnly = { ...combinedFilters, minPrice: undefined, maxPrice: undefined, category: undefined, brand: undefined };
        result = productService.getProducts(searchOnly);
      }
      if (result.products.length === 0) {
        // FINAL fallback: show all products
        result = productService.getProducts({ page: 1, limit: 24, sortBy: 'relevance' });
      }

      res.json({
        ...result,
        searchIntent,
        appliedFilters: aiFilters,
        suggestions: aiFilters.suggestions || [],
      });
    } catch (error) {
      console.error("Error in AI search:", error);
      res.status(500).json({ error: "AI search failed" });
    }
  });

  // Get search suggestions based on partial query
  app.get("/api/search/suggestions", async (req, res) => {
    try {
      const query = req.query.q as string;
      if (!query || query.length < 2) {
        return res.json({ suggestions: [] });
      }

      // Simple suggestions based on product names and categories
      const allProducts = productService.getProducts({ limit: 1000 }).products;
      const suggestions = new Set<string>();

      // Add matching product names
      allProducts.forEach(product => {
        if (product.name.toLowerCase().includes(query.toLowerCase())) {
          suggestions.add(product.name);
        }
        if (product.category?.toLowerCase().includes(query.toLowerCase())) {
          suggestions.add(product.category);
        }
      });

      // Add common search patterns
      const commonPatterns = [
        `${query} under ₹500`,
        `${query} under ₹225`,
        `promotional ${query}`,
        `custom ${query}`,
      ];

      commonPatterns.forEach(pattern => {
        if (pattern.toLowerCase().includes(query.toLowerCase())) {
          suggestions.add(pattern);
        }
      });

      res.json({
        suggestions: Array.from(suggestions).slice(0, 8)
      });
    } catch (error) {
      console.error("Error getting suggestions:", error);
      res.status(500).json({ error: "Failed to get suggestions" });
    }
  });

  // Vector-based product search endpoint
  app.post("/api/search/vector", async (req, res) => {
    try {
      const { query, filters, aiFilters } = req.body;
      let vectorResults;
      if (!query || query.trim() === "") {
        // Only use pagination filters if no search query
        vectorResults = await productService.vectorService.vectorSearch("", 100, { page: filters?.page, limit: filters?.limit });
      } else {
        vectorResults = await productService.vectorService.vectorSearch(query, 100, filters);
      }
      // Optionally, apply AI smart filters (e.g., feature, trending)
      let filtered = vectorResults;
      if (aiFilters && aiFilters.length > 0) {
        aiFilters.forEach(f => {
          if (f.type === 'feature') {
            filtered = filtered.filter(p =>
              (p.name && p.name.toLowerCase().includes(f.value.toLowerCase())) ||
              (p.description && p.description.toLowerCase().includes(f.value.toLowerCase()))
            );
          }
          // Add more AI filter types as needed
        });
      }
      // Pagination
      const page = filters?.page || 1;
      const limit = filters?.limit || 24;
      const total = filtered.length;
      const totalPages = Math.ceil(total / limit);
      const start = (page - 1) * limit;
      const products = filtered.slice(start, start + limit).map(r => r.product || r);
      res.json({ products, total, totalPages });
    } catch (error) {
      console.error("Error in vector search:", error);
      res.status(500).json({ error: "Vector search failed" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
