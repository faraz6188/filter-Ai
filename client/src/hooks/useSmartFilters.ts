import { useQuery } from "@tanstack/react-query";

export interface SmartFilter {
  id: string;
  label: string;
  type: 'category' | 'price' | 'feature';
  value: any;
  count: number;
  description: string;
}

export interface SmartFiltersResponse {
  suggestedFilters: SmartFilter[];
  trendingSearches: string[];
  personalizedSuggestions: string[];
}

export function useSmartFilters(query: string = "") {
  return useQuery<SmartFiltersResponse>({
    queryKey: ["/api/filters/smart", query],
    queryFn: async () => {
      const res = await fetch(`/api/filters/smart?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error("Failed to fetch smart filters");
      return res.json();
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 30 * 1000, // Refresh every 30 seconds for dynamic updates
  });
}