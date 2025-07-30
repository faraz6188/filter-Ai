import { useState, useEffect } from "react";
import { SearchHeader } from "../components/SearchHeader";
import { FilterSidebar } from "../components/FilterSidebar";
import { ProductGrid } from "../components/ProductGrid";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { useProducts, useAISearch, useVectorProducts } from "../hooks/useProducts";
import { useFilters } from "../hooks/useFilters";
import { useSmartFilters } from "../hooks/useSmartFilters";
import { Product } from "../types/product";
import { useToast } from "@/hooks/use-toast";
import { Filter } from "lucide-react";

export default function ProductCatalog() {
  const [searchQuery, setSearchQuery] = useState("");
  const [aiFilters, setAIFilters] = useState<any[]>([]);
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [cartCount, setCartCount] = useState(0);
  
  const { filters, updateFilter, clearFilters, setFilters } = useFilters();
  if (!filters.sortBy) {
    updateFilter('sortBy', 'price_high');
  }
  const { toast } = useToast();

  // Use vector search for all product fetching
  const { data: productsData, isLoading, error } = useVectorProducts(searchQuery, filters, aiFilters);
  const { data: smartFilters, isLoading: smartFiltersLoading } = useSmartFilters(searchQuery);

  // Debug: Log filters and query being sent to vector search
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log('Vector Search Query:', searchQuery, 'Filters:', filters, 'AI Filters:', aiFilters);
  }, [searchQuery, filters, aiFilters]);

  useEffect(() => {
    if (error) {
      toast({
        title: "Search Error",
        description: "Failed to load products. Please try again.",
        variant: "destructive",
      });
    }
  }, [error, toast]);

  const handleSearchChange = (query: string) => {
    setSearchQuery(query);
  };

  const handleSearchSubmit = (query: string) => {
    setSearchQuery(query);
    updateFilter('search', query);
  };

  const handleAddToCart = (product: Product) => {
    setCartCount(prev => prev + 1);
    toast({
      title: "Added to Cart",
      description: `${product.name} has been added to your cart.`,
    });
  };

  const handleToggleFavorite = (product: Product) => {
    toast({
      title: "Added to Favorites",
      description: `${product.name} has been added to your favorites.`,
    });
  };

  const handleFilterChange = (key: keyof typeof filters, value: any) => {
    updateFilter(key, value);
    if (key === 'search') {
      setSearchQuery(value || "");
    }
  };

  const handleClearFilters = () => {
    clearFilters();
    setSearchQuery("");
    setAIFilters([]);
  };

  // Handler for AI smart filter buttons
  const handleAISmartFilterClick = (filter: any) => {
    if (filter.type === 'category') {
      updateFilter('category', filter.value);
      setSearchQuery(filter.value); // Optionally set search bar
      updateFilter('search', filter.value); // Ensures vector search uses this
    } else if (filter.type === 'price') {
      if (filter.value.min !== undefined) updateFilter('minPrice', filter.value.min);
      if (filter.value.max !== undefined) updateFilter('maxPrice', filter.value.max);
      setSearchQuery(''); // Clear search bar for price-only filter
      updateFilter('search', '');
    } else if (filter.type === 'feature') {
      setSearchQuery(filter.value);
      updateFilter('search', filter.value);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <SearchHeader
        searchQuery={searchQuery}
        onSearchChange={handleSearchChange}
        onSearchSubmit={handleSearchSubmit}
        cartCount={cartCount}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="lg:grid lg:grid-cols-4 lg:gap-8">
          {/* Desktop Filter Sidebar */}
          <div className="hidden lg:block">
            <FilterSidebar
              filters={filters}
              onFilterChange={handleFilterChange}
              onClearFilters={handleClearFilters}
              setFilters={setFilters}
              onAISmartFilterClick={handleAISmartFilterClick}
            />
          </div>

          <div className="lg:col-span-3">
            <ProductGrid
              products={productsData?.products || []}
              total={productsData?.total || 0}
              totalPages={productsData?.totalPages || 0}
              currentPage={filters.page || 1}
              isLoading={isLoading}
              filters={filters}
              onFilterChange={handleFilterChange}
              onAddToCart={handleAddToCart}
              onToggleFavorite={handleToggleFavorite}
              onToggleMobileFilters={() => setMobileFiltersOpen(true)}
            />
          </div>

          {/* Mobile Filter Sheet */}
          <Sheet open={mobileFiltersOpen} onOpenChange={setMobileFiltersOpen}>
            <SheetContent side="left" className="w-80 p-0">
              <div className="h-full overflow-y-auto">
                <FilterSidebar
                  filters={filters}
                  onFilterChange={handleFilterChange}
                  onClearFilters={handleClearFilters}
                  setFilters={setFilters}
                  onAISmartFilterClick={handleAISmartFilterClick}
                  className="h-full"
                />
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </div>
  );
}
