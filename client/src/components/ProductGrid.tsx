import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Grid, List, Filter, Loader2 } from "lucide-react";
import { ProductCard } from "./ProductCard";
import { Product, ProductFilter } from "../types/product";
import { cn } from "@/lib/utils";

interface ProductGridProps {
  products: Product[];
  total: number;
  totalPages: number;
  currentPage: number;
  isLoading?: boolean;
  filters: ProductFilter;
  onFilterChange: (key: keyof ProductFilter, value: any) => void;
  onAddToCart?: (product: Product) => void;
  onToggleFavorite?: (product: Product) => void;
  onToggleMobileFilters?: () => void;
  className?: string;
}

export function ProductGrid({ 
  products, 
  total, 
  totalPages, 
  currentPage, 
  isLoading,
  filters,
  onFilterChange,
  onAddToCart,
  onToggleFavorite,
  onToggleMobileFilters,
  className 
}: ProductGridProps) {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  const handleSortChange = (value: string) => {
    onFilterChange('sortBy', value as ProductFilter['sortBy']);
  };

  const handlePageChange = (page: number) => {
    onFilterChange('page', page);
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages = [];
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    // Previous button
    pages.push(
      <Button
        key="prev"
        variant="outline"
        size="sm"
        disabled={currentPage === 1}
        onClick={() => handlePageChange(currentPage - 1)}
      >
        Previous
      </Button>
    );

    // First page
    if (startPage > 1) {
      pages.push(
        <Button
          key={1}
          variant={currentPage === 1 ? "default" : "outline"}
          size="sm"
          onClick={() => handlePageChange(1)}
        >
          1
        </Button>
      );
      if (startPage > 2) {
        pages.push(<span key="dots1" className="px-2">...</span>);
      }
    }

    // Visible pages
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <Button
          key={i}
          variant={currentPage === i ? "default" : "outline"}
          size="sm"
          onClick={() => handlePageChange(i)}
        >
          {i}
        </Button>
      );
    }

    // Last page
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        pages.push(<span key="dots2" className="px-2">...</span>);
      }
      pages.push(
        <Button
          key={totalPages}
          variant={currentPage === totalPages ? "default" : "outline"}
          size="sm"
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </Button>
      );
    }

    // Next button
    pages.push(
      <Button
        key="next"
        variant="outline"
        size="sm"
        disabled={currentPage === totalPages}
        onClick={() => handlePageChange(currentPage + 1)}
      >
        Next
      </Button>
    );

    return pages;
  };

  const LoadingSkeleton = () => (
    <div className={cn("grid gap-6", {
      "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3": viewMode === 'grid',
      "grid-cols-1": viewMode === 'list'
    })}>
      {[...Array(6)].map((_, i) => (
        <div key={i} className="bg-white rounded-lg shadow-sm border animate-pulse">
          <div className="h-48 bg-gray-200 rounded-t-lg"></div>
          <div className="p-4 space-y-3">
            <div className="h-4 bg-gray-200 rounded w-1/3"></div>
            <div className="h-4 bg-gray-200 rounded w-full"></div>
            <div className="h-3 bg-gray-200 rounded w-2/3"></div>
            <div className="flex justify-between items-center">
              <div className="h-6 bg-gray-200 rounded w-1/4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/6"></div>
            </div>
            <div className="h-8 bg-gray-200 rounded w-full"></div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className={cn("lg:col-span-3", className)}>
      {/* Results Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">
            {filters.search ? `Search Results for "${filters.search}"` : 'Products'}
          </h2>
          <p className="text-sm text-gray-600 mt-1">
            Showing {products.length === 0 ? 0 : ((currentPage - 1) * (filters.limit || 24)) + 1}-{Math.min(currentPage * (filters.limit || 24), total)} of {total.toLocaleString()} products
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <Select value={filters.sortBy || 'relevance'} onValueChange={handleSortChange}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="relevance">Sort by Relevance</SelectItem>
              <SelectItem value="price_low">Price: Low to High</SelectItem>
              <SelectItem value="price_high">Price: High to Low</SelectItem>
              <SelectItem value="newest">Newest First</SelectItem>
              <SelectItem value="rating">Best Rated</SelectItem>
            </SelectContent>
          </Select>
          
          <div className="hidden sm:flex border border-gray-300 rounded-md">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('grid')}
              className="rounded-r-none"
            >
              <Grid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('list')}
              className="rounded-l-none"
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Mobile Filter Toggle */}
          <Button
            variant="outline"
            size="sm"
            className="lg:hidden"
            onClick={onToggleMobileFilters}
          >
            <Filter className="h-4 w-4" />
            Filters
          </Button>
        </div>
      </div>

      {/* Loading Spinner */}
      {isLoading && (
        <div className="flex justify-center items-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}

      {/* Product Grid/List */}
      {isLoading ? (
        <LoadingSkeleton />
      ) : products.length === 0 ? (
        <div className="text-center py-12">
          <div className="max-w-md mx-auto">
            <h3 className="text-lg font-medium text-gray-900 mb-2">No products found</h3>
            <p className="text-gray-600 mb-4">
              Try adjusting your filters or search terms to find what you're looking for.
            </p>
            <Button 
              variant="outline" 
              onClick={() => {
                onFilterChange('search', '');
                onFilterChange('category', undefined);
                onFilterChange('brand', undefined);
                onFilterChange('minPrice', undefined);
                onFilterChange('maxPrice', undefined);
              }}
            >
              Clear all filters
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className={cn("grid gap-6 mb-8", {
            "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3": viewMode === 'grid',
            "grid-cols-1": viewMode === 'list'
          })}>
            {products.slice(0, 30).map((product) => (
              <ProductCard
                key={product.id}
                product={product}
                onAddToCart={onAddToCart}
                onToggleFavorite={onToggleFavorite}
                className={viewMode === 'list' ? 'flex-row' : ''}
                onBadgeClick={onFilterChange as (key: string, value: any) => void}
              />
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-8">
              <div className="flex items-center text-sm text-gray-700">
                <span>
                  Showing {((currentPage - 1) * (filters.limit || 24)) + 1} to {Math.min(currentPage * (filters.limit || 24), total)} of {total.toLocaleString()} results
                </span>
              </div>
              <div className="flex items-center space-x-2">
                {renderPagination()}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
