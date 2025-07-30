import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Filter, X, ChevronDown, Lightbulb, Brain, Sparkles, TrendingUp } from "lucide-react";
import { useFilterOptions } from "../hooks/useFilters";
import { useSmartFilters } from "../hooks/useSmartFilters";
import { ProductFilter } from "../types/product";
import { cn } from "@/lib/utils";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

interface FilterSidebarProps {
  filters: ProductFilter;
  onFilterChange: (key: keyof ProductFilter, value: any) => void;
  onClearFilters: () => void;
  setFilters: (newFilters: Partial<ProductFilter>) => void;
  className?: string;
  onAISmartFilterClick?: (filter: any) => void;
}

export function FilterSidebar({ filters, onFilterChange, onClearFilters, setFilters, className, onAISmartFilterClick }: FilterSidebarProps) {
  const { data: filterOptions, isLoading } = useFilterOptions();
  const [cacheBuster, setCacheBuster] = useState(0);
  const { data: smartFilters, isLoading: smartFiltersLoading, refetch: refetchSmartFilters } = useSmartFilters(`${filters.search || ""}#${cacheBuster}`);
  const [showMoreCategories, setShowMoreCategories] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const activeFiltersCount = Object.entries(filters).filter(([key, value]) => 
    value !== undefined && key !== 'page' && key !== 'limit' && (key !== 'sortBy' || value !== 'relevance')
  ).length;

  const handlePriceRangeChange = (min?: number, max?: number, checked?: boolean) => {
    if (checked) {
      onFilterChange('minPrice', min);
      onFilterChange('maxPrice', max === Infinity ? undefined : max);
    } else {
      // If unchecking and this was the active range, clear it
      if (filters.minPrice === min && (filters.maxPrice === max || (max === Infinity && !filters.maxPrice))) {
        onFilterChange('minPrice', undefined);
        onFilterChange('maxPrice', undefined);
      }
    }
  };

  const isPriceRangeActive = (min?: number, max?: number) => {
    return filters.minPrice === min && 
           (max === Infinity ? !filters.maxPrice : filters.maxPrice === max);
  };

  // Add sort options
  const sortOptions = [
    { value: 'relevance', label: 'Relevance' },
    { value: 'price_low', label: 'Price: Low to High' },
    { value: 'price_high', label: 'Price: High to Low' },
  ];

  // Function to refresh AI smart filters using react-query refetch
  const handleRefreshSmartFilters = async () => {
    setRefreshing(true);
    setCacheBuster(Date.now());
    setTimeout(() => setRefreshing(false), 500); // allow UI to show spinner
  };

  if (isLoading) {
    return (
      <div className={cn("lg:col-span-1", className)}>
        <Card className="sticky top-24">
          <CardContent className="p-6">
            <div className="animate-pulse space-y-4">
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-3 bg-gray-200 rounded"></div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className={cn("lg:col-span-1", className)}>
      <Card className="sticky top-24">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold text-gray-900 flex items-center">
              <Filter className="h-4 w-4 mr-2 text-primary" />
              Smart Filters
              {activeFiltersCount > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {activeFiltersCount}
                </Badge>
              )}
            </CardTitle>
            {activeFiltersCount > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClearFilters}
                className="text-primary hover:text-primary/80"
              >
                Clear All
              </Button>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* AI Smart Filters Section */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-gray-700 flex items-center">
                  <Brain className="h-4 w-4 mr-2 text-primary" />
                  AI Smart Filters
                </h3>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="bg-primary/10 text-primary text-xs">
                  <Sparkles className="h-3 w-3 mr-1" />
                  Live
                </Badge>
                <button
                  type="button"
                  className="p-1 rounded-full hover:bg-primary/10 transition"
                  title="Refresh AI Smart Filters"
                  onClick={handleRefreshSmartFilters}
                  disabled={refreshing}
                >
                  {refreshing ? (
                    <svg className="animate-spin w-5 h-5 text-primary" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-primary">
                      <path fillRule="evenodd" d="M10 3a7 7 0 100 14A7 7 0 0010 3zm3.707 5.293a1 1 0 00-1.414 0L10 10.586V7a1 1 0 10-2 0v6a1 1 0 001 1h6a1 1 0 100-2h-3.586l2.293-2.293a1 1 0 000-1.414z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
            {/* AI Smart Filters */}
            {!smartFiltersLoading && (smartFilters?.suggestedFilters?.length ?? 0) > 0 && (
              <div className="flex flex-wrap gap-2">
                {(smartFilters && smartFilters.suggestedFilters) ? smartFilters.suggestedFilters.map(filter => (
                  <button
                    key={filter.id}
                    className="px-3 py-1 rounded-full border-2 border-yellow-500 text-yellow-800 bg-yellow-100 hover:bg-yellow-200 text-xs font-medium transition shadow-sm"
                    onClick={() => {
                      if (onAISmartFilterClick) {
                        onAISmartFilterClick(filter);
                      } else {
                        if (filter.type === 'category') {
                          setFilters({ category: filter.value, search: filter.value });
                        } else if (filter.type === 'price') {
                          setFilters({
                            minPrice: filter.value.min,
                            maxPrice: filter.value.max,
                            search: ''
                          });
                        } else if (filter.type === 'feature') {
                          setFilters({ search: filter.value });
                        }
                      }
                    }}
                  >
                    {filter.label}
                  </button>
                )) : null}
              </div>
            )}
              </div>

          {/* Active Filter Chips */}
          {activeFiltersCount > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-700">Active Filters</h3>
              <div className="flex flex-wrap gap-2">
                {filters.category && (
                  <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
                    {filters.category}
                    <X 
                      className="h-3 w-3 ml-1 cursor-pointer" 
                      onClick={() => onFilterChange('category', undefined)}
                    />
                  </Badge>
                )}
                {filters.brand && (
                  <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
                    {filters.brand}
                    <X 
                      className="h-3 w-3 ml-1 cursor-pointer" 
                      onClick={() => onFilterChange('brand', undefined)}
                    />
                  </Badge>
                )}
                {(filters.minPrice !== undefined || filters.maxPrice !== undefined) && (
                  <Badge variant="outline" className="bg-accent/10 text-amber-700 border-accent/20">
                    ₹{filters.minPrice || 0} - {filters.maxPrice ? `₹${filters.maxPrice}` : 'Above'}
                    <X 
                      className="h-3 w-3 ml-1 cursor-pointer" 
                      onClick={() => {
                        onFilterChange('minPrice', undefined);
                        onFilterChange('maxPrice', undefined);
                      }}
                    />
                  </Badge>
                )}
                {filters.sortBy && filters.sortBy !== 'relevance' && (
                  <Badge variant="outline" className="bg-secondary/10 text-secondary border-secondary/20">
                    Sort: {(() => {
                      switch (filters.sortBy) {
                        case 'price_low': return 'Price: Low to High';
                        case 'price_high': return 'Price: High to Low';
                        case 'newest': return 'Newest First';
                        case 'rating': return 'Best Rated';
                        default: return 'Relevance';
                      }
                    })()}
                    <X 
                      className="h-3 w-3 ml-1 cursor-pointer" 
                      onClick={() => onFilterChange('sortBy', 'relevance')}
                    />
                  </Badge>
                )}
              </div>
              <Separator />
            </div>
          )}

          {/* Sort By */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700">Sort By</h3>
            <RadioGroup
              value={filters.sortBy || 'relevance'}
              onValueChange={(value) => {
                onFilterChange('sortBy', value);
                onFilterChange('search', value);
              }}
              className="flex flex-col gap-2"
            >
              {sortOptions.map(opt => (
                <label key={opt.value} className="flex items-center cursor-pointer">
                  <RadioGroupItem value={opt.value} />
                  <span className="ml-2 text-sm text-gray-600">{opt.label}</span>
                </label>
              ))}
            </RadioGroup>
          </div>

          <Separator />

          {/* Price Range (Radio) */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700">Price Range</h3>
            <RadioGroup
              value={(() => {
                if (filters.minPrice === undefined && filters.maxPrice === undefined) return '';
                return `${filters.minPrice ?? 0}-${filters.maxPrice ?? 'Infinity'}`;
              })()}
              onValueChange={(val) => {
                if (!val) {
                  onFilterChange('minPrice', undefined);
                  onFilterChange('maxPrice', undefined);
                  return;
                }
                const [min, max] = val.split('-');
                onFilterChange('minPrice', Number(min));
                onFilterChange('maxPrice', max === 'Infinity' ? undefined : Number(max));
                onFilterChange('search', `₹${min}${max !== 'Infinity' ? `–₹${max}` : '+'}`);
              }}
              className="flex flex-col gap-2"
            >
              {filterOptions?.priceRanges.map((range, index) => {
                let label = '';
                if (range.max === Infinity) label = `Above ₹${range.min}`;
                else if (range.min === 0) label = `₹0–₹${range.max}`;
                else label = `₹${range.min}–₹${range.max}`;
                return (
                  <label key={index} className="flex items-center cursor-pointer">
                    <RadioGroupItem value={`${range.min}-${range.max === Infinity ? 'Infinity' : range.max}`} />
                    <span className="ml-2 text-sm text-gray-600">{label}</span>
                    <span className="ml-2 text-xs text-gray-400">({range.count})</span>
                  </label>
                );
              })}
              <label className="flex items-center cursor-pointer">
                <RadioGroupItem value="" />
                <span className="ml-2 text-sm text-gray-600">All Prices</span>
              </label>
            </RadioGroup>
          </div>

          <Separator />

          {/* Categories */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700">Categories</h3>
            <div className="space-y-2">
              {filterOptions?.categories
                .slice(0, showMoreCategories ? undefined : 3)
                .map((category) => (
                <div key={category.name} className="flex items-center space-x-2">
                  <Checkbox
                    id={`category-${category.name}`}
                    checked={filters.category === category.name}
                    onCheckedChange={(checked) => {
                      onFilterChange('category', checked ? category.name : undefined);
                      if (checked) onFilterChange('search', category.name);
                    }}
                  />
                  <label
                    htmlFor={`category-${category.name}`}
                    className="text-sm text-gray-600 flex-1 cursor-pointer"
                  >
                    {category.name}
                  </label>
                  <span className="text-xs text-gray-400">({category.count})</span>
                </div>
              ))}
            </div>
            {filterOptions && filterOptions.categories.length > 3 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowMoreCategories(!showMoreCategories)}
                className="text-primary hover:text-primary/80 p-0 h-auto"
              >
                {showMoreCategories ? 'View Less' : 'View More'}
              </Button>
            )}
          </div>

          <Separator />

          {/* Brands (only if present) */}
          {filterOptions?.brands && filterOptions.brands.length > 0 && (
            <>
              <Separator />
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-gray-700">Brands</h3>
                <div className="space-y-2">
                  {filterOptions.brands.map((brand) => (
                    <div key={brand.name} className="flex items-center space-x-2">
                      <Checkbox
                        id={`brand-${brand.name}`}
                        checked={filters.brand === brand.name}
                        onCheckedChange={(checked) => {
                          onFilterChange('brand', checked ? brand.name : undefined);
                          if (checked) onFilterChange('search', brand.name);
                        }}
                      />
                      <label
                        htmlFor={`brand-${brand.name}`}
                        className="text-sm text-gray-600 flex-1 cursor-pointer"
                      >
                        {brand.name}
                      </label>
                      <span className="text-xs text-gray-400">({brand.count})</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* AI Insights */}
          <Card className="bg-primary/5 border-primary/10">
            <CardContent className="p-4">
              <div className="flex items-center mb-2">
                <Lightbulb className="h-4 w-4 text-primary mr-2" />
                <span className="text-sm font-medium text-primary">AI Insight</span>
              </div>
              <p className="text-xs text-gray-600">
                {filters.category === 'Drinkware' 
                  ? "Based on your search, customers often buy drinkware with promotional bags. Consider bundling for better deals!"
                  : "Popular items in this category are often purchased together with complementary products. Check out our bundle deals!"
                }
              </p>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
}
