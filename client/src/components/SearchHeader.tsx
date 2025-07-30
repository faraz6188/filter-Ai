import { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Search, Brain, Heart, ShoppingCart } from "lucide-react";
import { useSearchSuggestions } from "../hooks/useProducts";
import { cn } from "@/lib/utils";

interface SearchHeaderProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onSearchSubmit: (query: string) => void;
  cartCount?: number;
}

export function SearchHeader({ searchQuery, onSearchChange, onSearchSubmit, cartCount = 0 }: SearchHeaderProps) {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [inputValue, setInputValue] = useState(searchQuery);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  const { data: suggestions } = useSearchSuggestions(inputValue);

  useEffect(() => {
    setInputValue(searchQuery);
  }, [searchQuery]);

  const handleInputChange = (value: string) => {
    setInputValue(value);
    onSearchChange(value);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSearchSubmit(inputValue.trim());
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    onSearchSubmit(suggestion);
    setShowSuggestions(false);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        !inputRef.current?.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <header className="bg-white shadow-sm border-b sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-2xl font-bold text-primary">SmartCommerce</h1>
            </div>
          </div>

          {/* AI Search Bar */}
          <div className="flex-1 max-w-2xl mx-8 relative">
            <form onSubmit={handleSubmit}>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-gray-400" />
                </div>
                <Input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => handleInputChange(e.target.value)}
                  onFocus={() => setShowSuggestions(true)}
                  className="block w-full pl-10 pr-12 py-2 border border-gray-300 rounded-lg leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-2 focus:ring-primary focus:border-primary"
                  placeholder="Search with AI... Try 'blue mugs under ₹500' or 'promotional items for events'"
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                  <Badge variant="secondary" className="bg-primary/10 text-primary">
                    <Brain className="h-3 w-3 mr-1" />
                    AI
                  </Badge>
                </div>
              </div>
            </form>

            {/* AI Search Suggestions */}
            {showSuggestions && suggestions?.suggestions && suggestions.suggestions.length > 0 && (
              <div
                ref={suggestionsRef}
                className="absolute top-full left-0 right-0 bg-white border border-gray-200 rounded-lg shadow-lg mt-1 z-50"
              >
                <div className="p-2">
                  <div className="text-xs text-gray-500 mb-2">AI Suggestions</div>
                  <div className="space-y-1">
                    {suggestions.suggestions.map((suggestion, index) => (
                      <div
                        key={index}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="px-3 py-2 hover:bg-gray-50 rounded cursor-pointer text-sm flex items-center"
                      >
                        <Search className="h-3 w-3 mr-2 text-gray-400" />
                        {suggestion}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm">
              <Heart className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="relative">
              <ShoppingCart className="h-4 w-4" />
              {cartCount > 0 && (
                <span className="absolute -top-1 -right-1 h-4 w-4 bg-primary text-white rounded-full text-xs flex items-center justify-center">
                  {cartCount}
                </span>
              )}
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
}
