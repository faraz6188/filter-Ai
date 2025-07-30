import { useQuery } from "@tanstack/react-query";
import { ProductFilter, ProductsResponse, AISearchResponse } from "../types/product";

export function useProducts(filters: ProductFilter) {
  return useQuery<ProductsResponse>({
    queryKey: ["/api/products", filters],
    enabled: true,
  });
}

export function useVectorProducts(query: string, filters: any, aiFilters: any[]) {
  return useQuery<ProductsResponse>({
    queryKey: ["/api/search/vector", { query, filters, aiFilters }],
    queryFn: async () => {
      const res = await fetch("/api/search/vector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, filters, aiFilters }),
      });
      if (!res.ok) throw new Error("Vector search failed");
      return res.json();
    },
    enabled: true,
  });
}

export function useAISearch(query: string, filters?: any) {
  return useQuery<AISearchResponse>({
    queryKey: ["/api/search/ai", { query, filters }],
    enabled: !!query && query.length > 0,
    retry: false,
  });
}

export function useSearchSuggestions(query: string) {
  return useQuery<{ suggestions: string[] }>({
    queryKey: ["/api/search/suggestions", { q: query }],
    enabled: !!query && query.length >= 2,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}
