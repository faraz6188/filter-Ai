import * as fs from "fs";
import { GoogleGenAI } from "@google/genai";

// DON'T DELETE THIS COMMENT
// Follow these instructions when using this blueprint:
// - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
//   - do not change this unless explicitly requested by the user

// This API key is from Gemini Developer API Key, not vertex AI API Key
const ai = new GoogleGenAI({ apiKey: "AIzaSyBueCrxGbmPSZ9oFJXR6qrHWsUGIfsFY-0" });

export interface SearchIntent {
  intent: string;
  confidence: number;
  entities: string[];
}

export interface ProductFilters {
  category?: string;
  priceRange?: { min?: number; max?: number };
  brand?: string;
  sortBy?: string;
  suggestions?: string[];
}

export async function analyzeSearchIntent(query: string): Promise<SearchIntent> {
  try {
    const systemPrompt = `You are a search intent analyzer for an e-commerce product catalog. 
    Analyze the user's search query and determine their intent, confidence level, and extract relevant entities.
    
    Possible intents:
    - "product_search": looking for specific products
    - "category_browse": browsing a category
    - "price_filter": looking for products in a price range
    - "brand_search": looking for specific brand products
    - "feature_search": looking for products with specific features
    
    Return JSON in this format:
    {
      "intent": "intent_type",
      "confidence": 0.95,
      "entities": ["entity1", "entity2"]
    }`;

    const response = await ai.models.generateContent({
      model: "gemini-1.5-flash",
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: "object",
          properties: {
            intent: { type: "string" },
            confidence: { type: "number" },
            entities: { 
              type: "array",
              items: { type: "string" }
            },
          },
          required: ["intent", "confidence", "entities"],
        },
      },
      contents: query,
    });

    const rawJson = response.text;
    if (rawJson) {
      return JSON.parse(rawJson);
    } else {
      throw new Error("Empty response from model");
    }
  } catch (error) {
    console.error("Failed to analyze search intent:", error);
    return {
      intent: "product_search",
      confidence: 0.5,
      entities: [],
    };
  }
}

// Utility to generate a random seed for each call
function getRandomSeed() {
  return Math.floor(Math.random() * 100000);
}

export async function generateProductFilters(
  query: string,
  availableOptions: {
    categories: string[];
    brands: string[];
    priceBands?: string[];
    features?: string[];
  }
): Promise<ProductFilters> {
  try {
    const randomSeed = getRandomSeed();
    const systemPrompt = `You are an AI assistant for an e-commerce product catalog.
Given the following product data:
- Categories: ${availableOptions.categories.join(", ")}
- Brands: ${availableOptions.brands.join(", ")}
- Features: ${(availableOptions.features || []).join(", ")}

Generate 5–7 creative, relevant, and diverse filter suggestions for a product search/filter sidebar. Each suggestion should be:
- Based on the actual data above (not random, not test or placeholder values).
- Useful for helping a user find or discover products.
- For price filters, ONLY use labels like "Budget Friendly", "Cheap", "Affordable", or "Low Price" (do NOT use numeric price ranges or numbers in the label/value).
- Do NOT use any test, placeholder, or example values (e.g., no 'Test', 'Example', 'Sample', or numbers in the label/value).
- In this JSON format:
[
  {
    "label": "label for the filter",
    "type": "category|price|brand|feature",
    "value": "value for the filter",
    "description": "short description"
  },
  ...
]
Use this random seed to help generate different results: ${randomSeed}
`;

    const response = await ai.models.generateContent({
      model: "gemini-1.5-flash",
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: "object",
          properties: {
            category: { type: "string" },
            priceRange: {
              type: "object",
              properties: {
                min: { type: "number" },
                max: { type: "number" },
              },
            },
            brand: { type: "string" },
            sortBy: { type: "string" },
            suggestions: {
              type: "array",
              items: { type: "string" }
            },
          },
        },
      },
      contents: `Search query: "${query}"`,
    });

    const rawJson = response.text;
    let filters: ProductFilters = {};
    if (rawJson) {
      filters = JSON.parse(rawJson);
    }

    // --- Fuzzy category matching ---
    if (filters.category) {
      // Try to find the best match from available categories
      const lowerCat = filters.category.toLowerCase();
      const bestMatch = availableOptions.categories.find(cat =>
        cat.toLowerCase().includes(lowerCat) || lowerCat.includes(cat.toLowerCase())
      );
      if (bestMatch) {
        filters.category = bestMatch;
      } else {
        // Try to extract from query if not matched
        const queryMatch = availableOptions.categories.find(cat =>
          query.toLowerCase().includes(cat.toLowerCase())
        );
        if (queryMatch) filters.category = queryMatch;
        else delete filters.category;
      }
    } else {
      // No category from AI, try to extract from query
      const queryMatch = availableOptions.categories.find(cat =>
        query.toLowerCase().includes(cat.toLowerCase())
      );
      if (queryMatch) filters.category = queryMatch;
    }

    // --- Regex-based price extraction as fallback ---
    if (!filters.priceRange || (!filters.priceRange.min && !filters.priceRange.max)) {
      // under/below/less than X
      const underMatch = query.match(/under[\s₹$]*([\d,]+)/i) || query.match(/below[\s₹$]*([\d,]+)/i) || query.match(/less than[\s₹$]*([\d,]+)/i);
      if (underMatch) {
        filters.priceRange = { max: parseInt(underMatch[1].replace(/,/g, '')) };
      }
      // above/over/more than X
      const aboveMatch = query.match(/above[\s₹$]*([\d,]+)/i) || query.match(/over[\s₹$]*([\d,]+)/i) || query.match(/more than[\s₹$]*([\d,]+)/i);
      if (aboveMatch) {
        filters.priceRange = { min: parseInt(aboveMatch[1].replace(/,/g, '')) };
      }
      // between X and Y
      const betweenMatch = query.match(/between[\s₹$]*([\d,]+)[^\d]+([\d,]+)/i) || query.match(/([\d,]+)[\s-]+([\d,]+)/i);
      if (betweenMatch) {
        filters.priceRange = {
          min: parseInt(betweenMatch[1].replace(/,/g, '')),
          max: parseInt(betweenMatch[2].replace(/,/g, '')),
        };
      }
    }

    // --- Brand fuzzy match ---
    if (filters.brand) {
      const lowerBrand = filters.brand.toLowerCase();
      const bestBrand = availableOptions.brands.find(brand =>
        brand.toLowerCase().includes(lowerBrand) || lowerBrand.includes(brand.toLowerCase())
      );
      if (bestBrand) {
        filters.brand = bestBrand;
      } else {
        delete filters.brand;
      }
    }

    // --- Post-process suggestions to remove numeric/test/placeholder values ---
    if (filters.suggestions && Array.isArray(filters.suggestions)) {
      filters.suggestions = filters.suggestions.filter(s => {
        // Remove any suggestion with numbers or test/placeholder words
        return !(/\d/.test(s) || /test|example|sample|placeholder/i.test(s));
      });
    }

    return filters;
  } catch (error) {
    console.error("Failed to generate product filters:", error);
    return {
      suggestions: [
        "Try searching for specific product names",
        "Use price ranges like 'under ₹500'",
        "Browse by categories like 'drinkware' or 'bags'",
      ]
    };
  }
}

export async function generateSearchSuggestions(
  partialQuery: string,
  recentSearches: string[] = [],
  popularCategories: string[] = []
): Promise<string[]> {
  try {
    const systemPrompt = `You are an AI assistant helping users search for promotional products and custom merchandise.
    Generate helpful search suggestions based on the partial query.
    
    Recent searches: ${recentSearches.join(", ")}
    Popular categories: ${popularCategories.join(", ")}
    
    Generate 5-8 relevant search suggestions that would help users find products.
    Focus on practical, actionable suggestions.
    
    Return JSON array: ["suggestion1", "suggestion2", ...]`;

    const response = await ai.models.generateContent({
      model: "gemini-1.5-flash",
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: "array",
          items: { type: "string" }
        },
      },
      contents: `Partial query: "${partialQuery}"`,
    });

    const rawJson = response.text;
    if (rawJson) {
      return JSON.parse(rawJson);
    } else {
      return [];
    }
  } catch (error) {
    console.error("Failed to generate search suggestions:", error);
    return [
      `${partialQuery} under ₹500`,
      `promotional ${partialQuery}`,
      `custom ${partialQuery}`,
    ];
  }
}
