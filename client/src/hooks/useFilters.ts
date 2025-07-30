import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { ProductFilter, FilterOptions } from "../types/product";

export function useFilters() {
  const [filters, setFilters] = useState<ProductFilter>({
    page: 1,
    limit: 24,
    sortBy: 'relevance',
  });

  const updateFilter = useCallback((key: keyof ProductFilter, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value,
      page: key === 'page' ? value : 1, // Reset page when other filters change
    }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({
      page: 1,
      limit: 24,
      sortBy: 'relevance',
    });
  }, []);

  const toggleCategoryFilter = useCallback((category: string) => {
    setFilters(prev => ({
      ...prev,
      category: prev.category === category ? undefined : category,
      page: 1,
    }));
  }, []);

  const setPriceRange = useCallback((min?: number, max?: number) => {
    setFilters(prev => ({
      ...prev,
      minPrice: min,
      maxPrice: max,
      page: 1,
    }));
  }, []);

  const toggleBrandFilter = useCallback((brand: string) => {
    setFilters(prev => ({
      ...prev,
      brand: prev.brand === brand ? undefined : brand,
      page: 1,
    }));
  }, []);

  const batchSetFilters = useCallback((newFilters: Partial<ProductFilter>) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters,
      page: 1, // Reset page on filter change
    }));
  }, []);

  return {
    filters,
    updateFilter,
    clearFilters,
    toggleCategoryFilter,
    setPriceRange,
    toggleBrandFilter,
    setFilters: batchSetFilters,
  };
}

export function useFilterOptions() {
  return useQuery<FilterOptions>({
    queryKey: ["/api/filters"],
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}
