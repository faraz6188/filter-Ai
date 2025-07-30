import { GoogleGenAI } from "@google/genai";
import { Product } from "@shared/schema";

// This API key is from Gemini Developer API Key, not vertex AI API Key
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export interface ProductVector {
  id: number;
  embedding: number[];
  productData: Product;
  searchableText: string;
}

export interface VectorSearchResult {
  product: Product;
  similarity: number;
}

export class VectorService {
  private vectors: ProductVector[] = [];
  private isInitialized = false;

  async initialize(products: Product[]): Promise<void> {
    if (this.isInitialized) return;

    console.log('Initializing vector embeddings for products...');
    
    // Create vectors without using Gemini API to avoid rate limits
    // Use our simple vector generation for all products
    for (const product of products) {
      try {
        const searchableText = this.createSearchableText(product);
        const embedding = this.createSimpleVector(searchableText);
        
        this.vectors.push({
          id: product.id,
          embedding,
          productData: product,
          searchableText
        });
      } catch (error) {
        console.error(`Failed to create embedding for product ${product.id}:`, error);
      }
    }

    this.isInitialized = true;
    console.log(`Vector initialization complete. ${this.vectors.length} product embeddings created.`);
  }

  private createSearchableText(product: Product): string {
    // Combine all searchable fields into a single text
    const parts = [
      product.name,
      product.category,
      product.description,
      product.brand,
      product.originalText,
      // Add price context for better semantic search
      product.price ? `price ₹${product.price}` : '',
      // Add rating context
      product.rating ? `rated ${product.rating} stars` : ''
    ].filter(Boolean);

    return parts.join(' ').toLowerCase();
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [{
          role: "user",
          parts: [{ text: `Generate a semantic embedding for this product text: ${text}` }]
        }]
      });

      // Since Gemini doesn't have a dedicated embedding endpoint,
      // we'll create a simple hash-based vector for demonstration
      // In production, you'd use a proper embedding model like text-embedding-004
      return this.createSimpleVector(text);
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      return this.createSimpleVector(text);
    }
  }

  private createSimpleVector(text: string): number[] {
    // Create a simple 384-dimensional vector based on text content
    const vector = new Array(384).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    
    // Use word frequencies and positions to create meaningful vectors
    const wordMap = new Map<string, number>();
    words.forEach((word, index) => {
      if (word.length > 2) { // Skip very short words
        wordMap.set(word, (wordMap.get(word) || 0) + 1);
      }
    });

    // Convert words to vector dimensions using a simple hash
    Array.from(wordMap.entries()).forEach(([word, freq]) => {
      const hash = this.simpleHash(word);
      for (let i = 0; i < 3; i++) { // Use 3 dimensions per word
        const idx = (hash + i) % vector.length;
        vector[idx] += freq * (i + 1) * 0.1;
      }
    });

    // Normalize the vector
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  async vectorSearch(query: string, limit: number = 20, filters?: {
    category?: string;
    minPrice?: number;
    maxPrice?: number;
    brand?: string;
  }): Promise<VectorSearchResult[]> {
    if (!this.isInitialized) {
      throw new Error('Vector service not initialized');
    }

    // If a search query is present, do a fast substring match and return only matching products (no AI)
    if (query && query.length > 0) {
      const q = query.toLowerCase();
      const substringMatches = this.vectors
        .filter(v => {
          const name = v.productData.name?.toLowerCase() || "";
          const category = v.productData.category?.toLowerCase() || "";
          const desc = v.productData.description?.toLowerCase() || "";
          return (
            name.includes(q) ||
            category.includes(q) ||
            desc.includes(q)
          );
        })
        .slice(0, limit)
        .map(v => ({ product: v.productData, similarity: 1 }));
      return substringMatches;
    }
    // If no search query, return all products (up to limit)
    return this.vectors.slice(0, limit).map(v => ({ product: v.productData, similarity: 1 }));
  }

  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude > 0 ? dotProduct / magnitude : 0;
  }

  async generateSmartFilters(query: string): Promise<{
    suggestedCategories: string[];
    suggestedPriceRanges: { min?: number; max?: number; label: string }[];
    suggestedBrands: string[];
  }> {
    try {
      const systemPrompt = `You are an AI assistant for an e-commerce platform. 
      Based on the search query, suggest relevant filters that would help users find products.
      
      Analyze the query for:
      1. Category intentions (drinkware, bags, apparel, pens, tools, etc.)
      2. Price intentions (under 500, under 225, budget, premium, etc.)
      3. Brand preferences
      
      Return JSON in this format:
      {
        "suggestedCategories": ["category1", "category2"],
        "suggestedPriceRanges": [
          {"min": 0, "max": 50, "label": "Under ₹30"},
          {"min": 0, "max": 50, "label": "Under ₹50"}
        ],
        "suggestedBrands": ["brand1", "brand2"]
      }`;

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              suggestedCategories: {
                type: "array",
                items: { type: "string" }
              },
              suggestedPriceRanges: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    min: { type: "number" },
                    max: { type: "number" },
                    label: { type: "string" }
                  }
                }
              },
              suggestedBrands: {
                type: "array",
                items: { type: "string" }
              }
            }
          }
        },
        contents: `Search query: "${query}"`
      });

      const rawJson = response.text;
      if (rawJson) {
        return JSON.parse(rawJson);
      }
    } catch (error) {
      console.error('Failed to generate smart filters:', error);
    }

    // Fallback smart filters based on common patterns
    const fallbackFilters = {
      suggestedCategories: [] as string[],
      suggestedPriceRanges: [] as { min?: number; max?: number; label: string }[],
      suggestedBrands: [] as string[]
    };

    const queryLower = query.toLowerCase();
    
    // Detect price intentions
    if (queryLower.includes('under') || queryLower.includes('below') || queryLower.includes('budget')) {
      if (queryLower.includes('225')) {
        fallbackFilters.suggestedPriceRanges.push({ max: 225, label: "Under ₹225" });
      }
      if (queryLower.includes('500')) {
        fallbackFilters.suggestedPriceRanges.push({ max: 500, label: "Under ₹500" });
      }
    }

    // Detect category intentions
    if (queryLower.includes('mug') || queryLower.includes('cup') || queryLower.includes('bottle')) {
      fallbackFilters.suggestedCategories.push('Drinkware');
    }
    if (queryLower.includes('bag') || queryLower.includes('tote')) {
      fallbackFilters.suggestedCategories.push('Bags and totes');
    }
    if (queryLower.includes('pen') || queryLower.includes('pencil')) {
      fallbackFilters.suggestedCategories.push('Custom pens');
    }

    return fallbackFilters;
  }

  getVectorStats(): { totalVectors: number; isInitialized: boolean } {
    return {
      totalVectors: this.vectors.length,
      isInitialized: this.isInitialized
    };
  }
}