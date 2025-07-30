import fs from 'fs';
import path from 'path';
import { Product, InsertProduct } from '@shared/schema';
import { ImageService } from './imageService';
import { VectorService } from './vectorService';

export interface ProductData {
  id: number;
  name: string;
  category: string;
  url: string;
  original_text: string;
}

export interface ProcessedProduct extends Product {
  estimatedPrice?: number;
  extractedBrand?: string;
}

export class ProductService {
  private products: ProcessedProduct[] = [];
  private categories: Set<string> = new Set();
  private brands: Set<string> = new Set();
  private priceRanges: { min: number; max: number; count: number }[] = [];
  private vectorService: VectorService;

  constructor() {
    this.vectorService = new VectorService();
    this.loadProducts();
    this.initializeVectorSearch();
  }

  private async initializeVectorSearch() {
    // Initialize vector embeddings in background
    setTimeout(async () => {
      try {
        await this.vectorService.initialize(this.products);
        console.log('Vector search initialized successfully');
      } catch (error) {
        console.error('Failed to initialize vector search:', error);
      }
    }, 5000); // Wait 5 seconds after server start
  }

  private loadProducts() {
    try {
      // Try the newer file first, fallback to older one
      let dataPath = path.join(process.cwd(), 'attached_assets', 'product_data_1750678538573.json');
      if (!fs.existsSync(dataPath)) {
        dataPath = path.join(process.cwd(), 'attached_assets', 'product_data_1750678073988.json');
      }
      
      const rawData = fs.readFileSync(dataPath, 'utf8');
      const productData: ProductData[] = JSON.parse(rawData);

      this.products = productData.map(item => this.processProduct(item));
      this.buildFilterData();
      
      // Start image fetching process in background
      this.startImageFetching();
      
      console.log(`Loaded ${this.products.length} products`);
      console.log(`Categories: ${this.categories.size}`);
      console.log(`Brands: ${this.brands.size}`);
    } catch (error) {
      console.error('Error loading product data:', error);
      this.products = [];
    }
  }

  private async startImageFetching() {
    const batchSize = 10;
    const concurrency = 5;
    const productsWithUrls = this.products.filter(p => p.url && p.url !== 'nan');
    console.log(`Fetching images for ${productsWithUrls.length} products (one-time batch)...`);
    await this.processBatch(productsWithUrls, batchSize, concurrency);
    console.log('Image fetching complete.');
  }
  
  private async processBatch(products: ProcessedProduct[], batchSize: number, concurrency: number) {
    for (let i = 0; i < products.length; i += batchSize) {
      const batch = products.slice(i, i + batchSize);
      // Process batch with controlled concurrency
      const semaphore = this.createSemaphore(concurrency);
      const promises = batch.map(async (product) => {
        return semaphore(async () => {
          try {
            const imageResult = await ImageService.extractImageFromUrl(product.url!);
            if (imageResult.success && imageResult.imageUrl) {
              product.imageUrl = imageResult.imageUrl;
              // console.log(`✓ Image found for product ${product.id}: ${product.name}`); // Suppressed terminal output
              return true;
            }
            return false;
          } catch (error) {
            // console.warn(`⚠️ Failed to extract image for product ${product.id}: ${error}`); // Suppressed terminal output
            return false;
          }
        });
      });
      await Promise.allSettled(promises);
      // No delay between batches
    }
  }
  
  private createSemaphore(maxConcurrency: number) {
    let current = 0;
    const queue: (() => void)[] = [];
    
    return async <T>(fn: () => Promise<T>): Promise<T> => {
      return new Promise((resolve, reject) => {
        const execute = async () => {
          current++;
          try {
            const result = await fn();
            resolve(result);
          } catch (error) {
            reject(error);
          } finally {
            current--;
            if (queue.length > 0) {
              const next = queue.shift()!;
              next();
            }
          }
        };
        
        if (current < maxConcurrency) {
          execute();
        } else {
          queue.push(execute);
        }
      });
    };
  }

  private processProduct(item: ProductData): ProcessedProduct {
    // Extract price from original text or name
    const priceMatch = item.original_text?.match(/₹?(\d+(?:,\d+)*(?:\.\d{2})?)/);
    const estimatedPrice = priceMatch ? parseFloat(priceMatch[1].replace(/,/g, '')) : this.estimatePrice(item.category);

    // Extract brand from name or original text
    const extractedBrand = this.extractBrand(item.name, item.original_text);

    // Clean category
    const cleanCategory = this.cleanCategory(item.category);

    // Generate description from original text
    const description = this.extractDescription(item.original_text);

    // For now, use fallback images - we'll implement async image fetching separately
    const imageUrl = this.generateImageUrl(item.name, cleanCategory);

    const product: ProcessedProduct = {
      id: item.id,
      name: item.name,
      category: cleanCategory,
      url: item.url,
      originalText: item.original_text,
      price: estimatedPrice.toString(),
      imageUrl,
      description,
      brand: extractedBrand,
      rating: this.generateRating().toString(),
      reviewCount: Math.floor(Math.random() * 500) + 10,
      estimatedPrice: estimatedPrice,
      extractedBrand
    };

    // Add to filter data
    if (cleanCategory && cleanCategory !== 'nan') {
      this.categories.add(cleanCategory);
    }
    if (extractedBrand) {
      this.brands.add(extractedBrand);
    }

    return product;
  }

  private cleanCategory(category: string): string {
    if (!category || category === 'nan') return '';
    
    // Remove "- l1" suffixes and clean up
    return category
      .replace(/\s*-\s*l1\s*$/i, '')
      .replace(/,.*$/, '') // Take first category if multiple
      .trim()
      .toLowerCase()
      .replace(/^\w/, c => c.toUpperCase()); // Capitalize first letter
  }

  private extractBrand(name: string, originalText: string): string {
    // Common promotional product brands
    const commonBrands = ['Myron', 'PromoShop', 'CustomCraft', 'BrandMax', 'PromoGear'];
    
    // Try to extract from name or text
    const text = `${name} ${originalText}`.toLowerCase();
    
    for (const brand of commonBrands) {
      if (text.includes(brand.toLowerCase())) {
        return brand;
      }
    }

    // Default brand based on category or random selection
    return commonBrands[Math.floor(Math.random() * commonBrands.length)];
  }

  private extractDescription(originalText: string): string {
    if (!originalText || originalText === 'nan') return '';
    
    // Extract meaningful description from original text
    const parts = originalText.split('. ');
    const descriptionParts = parts.filter(part => 
      part.length > 20 && 
      !part.includes('url:') && 
      !part.includes('category:') &&
      !part.includes('item size:')
    );
    
    return descriptionParts.slice(0, 2).join('. ').substring(0, 200) + '...';
  }

  private estimatePrice(category: string): number {
    const priceRanges: { [key: string]: [number, number] } = {
      'drinkware': [150, 800],
      'bags and totes': [200, 1200],
      'apparel and accessories': [300, 1500],
      'custom pens': [50, 300],
      'flashlights': [400, 1000],
      'default': [100, 600]
    };

    const key = category?.toLowerCase() || 'default';
    const range = priceRanges[key] || priceRanges['default'];
    
    return Math.floor(Math.random() * (range[1] - range[0] + 1)) + range[0];
  }

  private generateImageUrl(name: string, category: string): string {
    // Generate relevant image URLs based on product type and name
    const imageMap: { [key: string]: string } = {
      'drinkware': 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300&q=80',
      'bags and totes': 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300&q=80',
      'apparel and accessories': 'https://images.unsplash.com/photo-1516762689617-e1cffcef479d?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300&q=80',
      'custom pens': 'https://images.unsplash.com/photo-1455390582262-044cdead277a?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300&q=80',
      'flashlights': 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300&q=80',
    };

    const categoryKey = category.toLowerCase();
    return imageMap[categoryKey] || 'https://images.unsplash.com/photo-1560472355-536de3962603?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300&q=80';
  }

  private generateRating(): number {
    // Generate realistic ratings between 3.5 and 5.0
    return Math.round((Math.random() * 1.5 + 3.5) * 10) / 10;
  }

  private buildFilterData() {
    // Build price ranges
    let prices = this.products.map(p => p.estimatedPrice || 0).sort((a, b) => a - b);
    if (prices.length === 0) {
      this.priceRanges = [];
      return;
    }
    // Use 99th percentile as max price to avoid outliers
    const minPrice = prices[0];
    const p99Index = Math.floor(prices.length * 0.99);
    const maxPrice = prices[p99Index];
    const bandCount = 4; // Number of bands
    const step = Math.ceil((maxPrice - minPrice) / bandCount) || 1;
    this.priceRanges = [];
    for (let i = 0; i < bandCount; i++) {
      const bandMin = minPrice + i * step;
      const bandMax = i === bandCount - 1 ? Infinity : minPrice + (i + 1) * step;
      const count = prices.filter(p => p >= bandMin && (bandMax === Infinity ? true : p < bandMax)).length;
      this.priceRanges.push({ min: Math.floor(bandMin), max: bandMax === Infinity ? Infinity : Math.ceil(bandMax), count });
    }
    // Debug log
    console.log('Price bands:', this.priceRanges, 'Min:', minPrice, '99th percentile Max:', maxPrice);
  }

  getProducts(filters: {
    category?: string;
    minPrice?: number;
    maxPrice?: number;
    brand?: string;
    search?: string;
    minRating?: number;
    page?: number;
    limit?: number;
    sortBy?: string;
  } = {}): { products: ProcessedProduct[]; total: number; totalPages: number } {
    let filtered = [...this.products];

    // Apply filters
    if (filters.category) {
      filtered = filtered.filter(p => 
        p.category?.toLowerCase().includes(filters.category!.toLowerCase())
      );
    }

    if (filters.brand) {
      filtered = filtered.filter(p => 
        p.brand?.toLowerCase().includes(filters.brand!.toLowerCase())
      );
    }

    if (filters.minPrice !== undefined) {
      filtered = filtered.filter(p => (p.estimatedPrice || 0) >= filters.minPrice!);
    }

    if (filters.maxPrice !== undefined) {
      filtered = filtered.filter(p => (p.estimatedPrice || 0) <= filters.maxPrice!);
    }

    if (filters.minRating !== undefined) {
      filtered = filtered.filter(p => parseFloat(p.rating || '0') >= filters.minRating!);
    }

    if (filters.search) {
      const searchTerm = filters.search.toLowerCase();
      filtered = filtered.filter(p => 
        p.name.toLowerCase().includes(searchTerm) ||
        p.description?.toLowerCase().includes(searchTerm) ||
        p.category?.toLowerCase().includes(searchTerm)
      );
    }

    // Apply sorting
    this.sortProducts(filtered, filters.sortBy || 'relevance');

    // Apply pagination
    const page = filters.page || 1;
    const limit = filters.limit || 24;
    const total = filtered.length;
    const totalPages = Math.ceil(total / limit);
    const start = (page - 1) * limit;
    const products = filtered.slice(start, start + limit);

    return { products, total, totalPages };
  }

  private sortProducts(products: ProcessedProduct[], sortBy: string) {
    switch (sortBy) {
      case 'price_low':
        products.sort((a, b) => (a.estimatedPrice || 0) - (b.estimatedPrice || 0));
        break;
      case 'price_high':
        products.sort((a, b) => (b.estimatedPrice || 0) - (a.estimatedPrice || 0));
        break;
      case 'rating':
        products.sort((a, b) => parseFloat(b.rating || '0') - parseFloat(a.rating || '0'));
        break;
      case 'newest':
        products.sort((a, b) => b.id - a.id);
        break;
      default: // relevance
        // Keep original order or apply relevance scoring
        break;
    }
  }

  getCategories(): { name: string; count: number }[] {
    const categoryCounts: { [key: string]: number } = {};
    
    this.products.forEach(product => {
      if (product.category && product.category !== 'nan') {
        categoryCounts[product.category] = (categoryCounts[product.category] || 0) + 1;
      }
    });

    return Object.entries(categoryCounts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);
  }

  getBrands(): { name: string; count: number }[] {
    const brandCounts: { [key: string]: number } = {};
    
    this.products.forEach(product => {
      if (product.brand) {
        brandCounts[product.brand] = (brandCounts[product.brand] || 0) + 1;
      }
    });

    return Object.entries(brandCounts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);
  }

  getPriceRanges(): { min: number; max: number; count: number }[] {
    return this.priceRanges;
  }

  // Helper to replace placeholders with real product data
  private replacePlaceholdersWithProductData(label: string): string {
    const match = label.match(/(\d{3,})/);
    if (!match) return label;
    const id = parseInt(match[1], 10);
    const product = this.products.find((p: ProcessedProduct) => p.id === id);
    console.log('Replacing placeholder:', label, 'with product:', product);
    if (product) {
      if (/category/i.test(label)) {
        return product.category || label;
      }
      if (/brand/i.test(label)) {
        return product.brand || label;
      }
      if (/suggestion/i.test(label)) {
        return product.name || label;
      }
    } else if (this.products.length > 0) {
      // Fallback: pick a random product
      const randomProduct = this.products[Math.floor(Math.random() * this.products.length)];
      if (/category/i.test(label)) {
        return randomProduct.category || label;
      }
      if (/brand/i.test(label)) {
        return randomProduct.brand || label;
      }
      if (/suggestion/i.test(label)) {
        return randomProduct.name || label;
      }
    }
    return label;
  }

  async getAISmartFilters(query: string = ""): Promise<{
    suggestedFilters: Array<{
      id: string;
      label: string;
      type: 'category' | 'price' | 'feature';
      value: any;
      count: number;
      description: string;
    }>;
    trendingSearches: string[];
    personalizedSuggestions: string[];
  }> {
    try {
      const categories = this.getCategories().map(c => c.name);
      const brands = this.getBrands().map(b => b.name);
      // Only pass categories and brands to Gemini for creative suggestions
      const features = ['customizable', 'eco-friendly', 'trending', 'bestseller'];
      const aiFilters = await import("./gemini").then(m =>
        m.generateProductFilters(query, { categories, brands, features })
      );
      console.log("Gemini AI filters response:", aiFilters);

      // Replace placeholders in aiFilters fields
      if (aiFilters.category && /(\d{3,})/.test(aiFilters.category)) {
        aiFilters.category = this.replacePlaceholdersWithProductData(aiFilters.category);
      }
      if (aiFilters.brand && /(\d{3,})/.test(aiFilters.brand)) {
        aiFilters.brand = this.replacePlaceholdersWithProductData(aiFilters.brand);
      }
      if (aiFilters.suggestions && Array.isArray(aiFilters.suggestions)) {
        aiFilters.suggestions = aiFilters.suggestions.map((s: string) => this.replacePlaceholdersWithProductData(s));
      }

      // Add a layer: parse Gemini JSON, randomly select 'value' or 'description' for label
      const aiSmartFilters = (aiFilters.suggestions || []).map((s: string, i: number) => {
        try {
          const obj = JSON.parse(s);
          const fields = ['value', 'description'] as const;
          const chosenField = fields[Math.floor(Math.random() * fields.length)];
          // Only use the chosen field as the label, nothing else
          return {
            id: `ai-suggestion-${i}`,
            label: String(obj[chosenField]), // Only value or description
            type: 'feature' as 'feature',
            value: obj.value,
            count: 0,
            description: ''
          };
        } catch (e) {
          // Fallback: show a generic label if parsing fails
          return {
            id: `ai-suggestion-${i}`,
            label: 'AI Suggestion',
            type: 'feature' as 'feature',
            value: '',
            count: 0,
            description: ''
          };
        }
      });

      if (aiSmartFilters.length === 0) {
        console.log("Falling back to static filters because Gemini returned no suggestions.");
        return this.getFallbackSmartFilters();
      }
      return {
        suggestedFilters: aiSmartFilters,
        trendingSearches: aiFilters.suggestions || [],
        personalizedSuggestions: [
          'Based on AI analysis',
          'Frequently bought together',
          'New arrivals this week'
        ]
      };
    } catch (error) {
      console.error('Failed to generate AI smart filters:', error);
      return this.getFallbackSmartFilters();
    }
  }

  private getFallbackSmartFilters() {
    return {
      suggestedFilters: [
        {
          id: 'popular-drinkware',
          label: 'Popular Drinkware',
          type: 'category' as const,
          value: 'Drinkware',
          count: this.products.filter(p => p.category?.toLowerCase().includes('drinkware')).length,
          description: 'Mugs, tumblers & bottles'
        },
        {
          id: 'budget-picks',
          label: 'Budget Picks',
          type: 'price' as const,
          value: { max: 225 },
          count: this.products.filter(p => parseFloat(p.price || '0') <= 225).length,
          description: 'Great value under ₹225'
        }
      ],
      trendingSearches: [
        'custom mugs under ₹500',
        'promotional bags',
        'budget drinkware'
      ],
      personalizedSuggestions: [
        'Based on popular choices',
        'Trending this week'
      ]
    };
  }
}
