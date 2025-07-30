import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Heart, Star, ShoppingCart } from "lucide-react";
import { Product } from "../types/product";
import { cn } from "@/lib/utils";

interface ProductCardProps {
  product: Product;
  onAddToCart?: (product: Product) => void;
  onToggleFavorite?: (product: Product) => void;
  className?: string;
  onBadgeClick?: (key: string, value: any) => void;
}

export function ProductCard({ product, onAddToCart, onToggleFavorite, className, onBadgeClick }: ProductCardProps) {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const [isFavorite, setIsFavorite] = useState(false);

  const handleImageError = () => {
    setImageError(true);
    setImageLoading(false);
  };

  const handleImageLoad = () => {
    setImageLoading(false);
  };

  const handleAddToCart = () => {
    onAddToCart?.(product);
  };

  const handleToggleFavorite = () => {
    setIsFavorite(!isFavorite);
    onToggleFavorite?.(product);
  };

  const formatPrice = (price?: string) => {
    if (!price) return '₹0';
    const numPrice = parseFloat(price);
    return `₹${numPrice.toLocaleString('en-IN')}`;
  };

  const getCategoryBadgeColor = (category?: string) => {
    if (!category) return 'bg-gray-100 text-gray-800';
    
    const colors: { [key: string]: string } = {
      'drinkware': 'bg-primary/10 text-primary',
      'bags and totes': 'bg-green-100 text-green-800',
      'apparel and accessories': 'bg-purple-100 text-purple-800',
      'custom pens': 'bg-orange-100 text-orange-800',
      'flashlights': 'bg-gray-100 text-gray-800',
    };
    
    return colors[category.toLowerCase()] || 'bg-gray-100 text-gray-800';
  };

  const getProductBadge = (product: Product) => {
    const price = parseFloat(product.price || '0');
    const rating = parseFloat(product.rating || '0');
    
    if (price <= 225) return { text: 'Budget', color: 'bg-green-500 text-white' };
    if (rating >= 4.7) return { text: '', color: 'bg-accent text-white' };
    if (product.id % 10 === 0) return { text: 'New', color: 'bg-blue-500 text-white' };
    return null;
  };

  const badge = getProductBadge(product);
  const fallbackImage = "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?ixlib=rb-4.0.3&w=400&h=300&fit=crop";
  
  // Use unique fallback images based on product category if main image fails
  const getCategoryFallbackImage = (category?: string) => {
    const categoryImages: { [key: string]: string } = {
      'drinkware': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?ixlib=rb-4.0.3&w=400&h=300&fit=crop',
      'bags and totes': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&w=400&h=300&fit=crop',
      'apparel and accessories': 'https://images.unsplash.com/photo-1516762689617-e1cffcef479d?ixlib=rb-4.0.3&w=400&h=300&fit=crop',
      'custom pens': 'https://images.unsplash.com/photo-1455390582262-044cdead277a?ixlib=rb-4.0.3&w=400&h=300&fit=crop',
      'flashlights': 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?ixlib=rb-4.0.3&w=400&h=300&fit=crop',
    };
    return categoryImages[category?.toLowerCase() || ''] || fallbackImage;
  };

  return (
    <Card className={cn("group hover:shadow-md transition-shadow duration-200", className)}>
      <div className="relative">
        {/* Product Image */}
        <div className="relative h-48 overflow-hidden rounded-t-lg bg-gray-100">
          {imageLoading && (
            <div className="absolute inset-0 bg-gray-200 animate-pulse" />
          )}
          <img
            src={imageError ? getCategoryFallbackImage(product.category) : product.imageUrl || getCategoryFallbackImage(product.category)}
            alt={product.name}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
            onError={handleImageError}
            onLoad={handleImageLoad}
          />
        </div>

        {/* Favorite Button */}
        <Button
          variant="ghost"
          size="sm"
          className="absolute top-3 right-3 p-2 bg-white/80 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-white"
          onClick={handleToggleFavorite}
        >
          <Heart className={cn("h-4 w-4", {
            "fill-red-500 text-red-500": isFavorite,
            "text-gray-400 hover:text-red-500": !isFavorite
          })} />
        </Button>

        {/* Product Badge */}
        {badge && (
          <div className="absolute top-3 left-3">
            <Badge className={badge.color}>
              {badge.text}
            </Badge>
          </div>
        )}
      </div>

      <CardContent className="p-4">
        {/* Category Badge */}
        {product.category && (
          <div className="mb-2">
            <Badge 
              variant="secondary" 
              className={getCategoryBadgeColor(product.category) + ' cursor-pointer'}
              onClick={() => onBadgeClick && onBadgeClick('category', product.category)}
              title="Filter by this category"
            >
              {product.category}
            </Badge>
          </div>
        )}

        {/* Brand Badge */}
        {product.brand && (
          <div className="mb-2">
            <Badge 
              variant="secondary" 
              className="bg-blue-100 text-blue-800 cursor-pointer"
              onClick={() => onBadgeClick && onBadgeClick('brand', product.brand)}
              title="Filter by this brand"
            >
              {product.brand}
            </Badge>
          </div>
        )}

        {/* Product Name (click to search) */}
        <h3 
          className="text-sm font-medium text-gray-900 mb-2 line-clamp-2 min-h-[2.5rem] cursor-pointer hover:underline"
          title="Search for this product name"
          onClick={() => onBadgeClick && onBadgeClick('search', product.name)}
        >
          {product.name}
        </h3>

        {/* Product Description (click to search) */}
        {product.description && (
          <p 
            className="text-xs text-gray-600 mb-3 line-clamp-2 min-h-[2rem] cursor-pointer hover:underline"
            title="Search for this description"
            onClick={() => onBadgeClick && onBadgeClick('search', product.description)}
          >
            {product.description}
          </p>
        )}

        {/* Price and Rating */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center">
            <span 
              className="text-lg font-semibold text-gray-900 cursor-pointer hover:underline"
              title="Filter by this price range"
              onClick={() => {
                const price = parseFloat(product.price || '0');
                if (onBadgeClick) {
                  onBadgeClick('minPrice', Math.floor(price));
                  onBadgeClick('maxPrice', Math.ceil(price));
                }
              }}
            >
              {formatPrice(product.price)}
            </span>
            {product.estimatedPrice && product.estimatedPrice > parseFloat(product.price || '0') && (
              <span className="text-sm text-gray-500 line-through ml-1">
                ₹{Math.round(product.estimatedPrice * 1.3).toLocaleString('en-IN')}
              </span>
            )}
          </div>
          
          {product.rating && (
            <div 
              className="flex items-center text-sm text-yellow-500 cursor-pointer hover:underline"
              title="Filter by this rating or higher"
              onClick={() => {
                if (typeof product.rating === 'string') {
                  onBadgeClick && onBadgeClick('minRating', parseFloat(product.rating));
                }
              }}
            >
              <Star className="h-3 w-3 fill-current" />
              <span className="ml-1 text-gray-600">
                {product.rating} ({product.reviewCount || 0})
              </span>
            </div>
          )}
        </div>

        {/* Add to Cart Button */}
        <Button
          onClick={handleAddToCart}
          className="w-full bg-primary text-white hover:bg-primary/90 transition-colors text-sm font-medium"
        >
          <ShoppingCart className="h-4 w-4 mr-2" />
          Add to Cart
        </Button>
      </CardContent>
    </Card>
  );
}
