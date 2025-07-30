export interface Product {
  id: number;
  name: string;
  category?: string;
  url?: string;
  originalText?: string;
  price?: string;
  imageUrl?: string;
  description?: string;
  brand?: string;
  rating?: string;
  reviewCount?: number;
  estimatedPrice?: number;
  extractedBrand?: string;
}

export interface ProductFilter {
  category?: string;
  minPrice?: number;
  maxPrice?: number;
  brand?: string;
  search?: string;
  minRating?: number;
  page?: number;
  limit?: number;
  sortBy?: 'relevance' | 'price_low' | 'price_high' | 'newest' | 'rating';
}

export interface FilterOptions {
  categories: { name: string; count: number }[];
  brands: { name: string; count: number }[];
  priceRanges: { min: number; max: number; count: number }[];
}

export interface ProductsResponse {
  products: Product[];
  total: number;
  totalPages: number;
}

export interface AISearchResponse extends ProductsResponse {
  searchIntent: {
    intent: string;
    confidence: number;
    entities: string[];
  };
  appliedFilters: {
    category?: string;
    priceRange?: { min?: number; max?: number };
    brand?: string;
    sortBy?: string;
    suggestions?: string[];
  };
  suggestions: string[];
}

export interface SearchSuggestion {
  suggestions: string[];
}
