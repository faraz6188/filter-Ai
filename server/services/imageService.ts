import axios from 'axios';
import * as cheerio from 'cheerio';

export interface ImageExtractionResult {
  imageUrl?: string;
  success: boolean;
  error?: string;
}

export class ImageService {
  private static cache = new Map<string, string>();
  private static readonly TIMEOUT = 10000; // 10 seconds
  private static readonly MAX_RETRIES = 2;

  static async extractImageFromUrl(productUrl: string): Promise<ImageExtractionResult> {
    if (!productUrl || productUrl === 'nan') {
      return { success: false, error: 'Invalid URL' };
    }

    // Check cache first
    const cacheKey = productUrl;
    if (this.cache.has(cacheKey)) {
      return { success: true, imageUrl: this.cache.get(cacheKey) };
    }

    // Ensure URL has protocol
    let fullUrl = productUrl;
    if (!fullUrl.startsWith('http://') && !fullUrl.startsWith('https://')) {
      fullUrl = `https://${fullUrl}`;
    }

    for (let attempt = 0; attempt < this.MAX_RETRIES; attempt++) {
      try {
        const response = await axios.get(fullUrl, {
          timeout: this.TIMEOUT,
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
          },
        });

        const $ = cheerio.load(response.data);
        
        // Enhanced selectors with higher priority for product images
        const imageSelectors = [
          // High priority selectors
          'img[data-testid*="product"]',
          'img[id*="product"]',
          'img[class*="product-image"]',
          'img[class*="ProductImage"]',
          'img[class*="product_image"]',
          '.product-image img',
          '.ProductImage img',
          '.product_image img',
          '.product-gallery img',
          '.product-photos img',
          '.product-detail img',
          
          // Medium priority selectors
          'img[alt*="product"]',
          'img[src*="product"]',
          'img[data-src*="product"]',
          '.gallery img',
          '.main-image img',
          '.hero-image img',
          'main img',
          'article img',
          
          // Lower priority selectors
          'img[src*="upload"]',
          'img[src*="cdn"]',
          'img[src*="media"]',
          '.content img',
          '.container img'
        ];

        let imageUrl: string | undefined;
        let bestScore = 0;

        // Score-based image selection
        for (const selector of imageSelectors) {
          $(selector).each((_, element) => {
            const img = $(element);
            const src = img.attr('src') || img.attr('data-src') || img.attr('data-lazy-src') || img.attr('data-original');
            
            if (src && this.isValidImageUrl(src)) {
              const score = ImageService.scoreImage(src, img.attr('alt') || '', img.attr('class') || '');
              if (score > bestScore) {
                bestScore = score;
                imageUrl = this.resolveImageUrl(src, fullUrl);
              }
            }
          });
        }

        // If no high-scoring image found, use fallback approach
        if (!imageUrl || bestScore < 3) {
          $('img').each((_, element) => {
            const img = $(element);
            const src = img.attr('src') || img.attr('data-src');
            if (src && this.isValidImageUrl(src)) {
              const resolvedUrl = this.resolveImageUrl(src, fullUrl);
              if (this.isProductImage(src, img.attr('alt') || '')) {
                imageUrl = resolvedUrl;
                return false; // break
              }
            }
          });
        }

        if (imageUrl) {
          this.cache.set(cacheKey, imageUrl);
          return { success: true, imageUrl };
        }

        return { success: false, error: 'No suitable image found' };

      } catch (error) {
        if (attempt === this.MAX_RETRIES - 1) {
          return { 
            success: false, 
            error: error instanceof Error ? error.message : 'Unknown error' 
          };
        }
        // Progressive backoff
        await new Promise(resolve => setTimeout(resolve, (attempt + 1) * 1000));
      }
    }

    return { success: false, error: 'Max retries exceeded' };
  }

  private static isValidImageUrl(src: string): boolean {
    if (!src || src.length < 4) return false;
    
    const lowerSrc = src.toLowerCase();
    return (
      lowerSrc.includes('.jpg') ||
      lowerSrc.includes('.jpeg') ||
      lowerSrc.includes('.png') ||
      lowerSrc.includes('.webp') ||
      lowerSrc.includes('.gif')
    );
  }

  private static scoreImage(src: string, alt: string, className: string): number {
    const lowerSrc = src.toLowerCase();
    const lowerAlt = alt.toLowerCase();
    const lowerClass = className.toLowerCase();
    let score = 0;

    // High value indicators
    const highValuePatterns = [
      'product-image', 'product_image', 'productimage', 'main-image', 'hero-image'
    ];
    for (const pattern of highValuePatterns) {
      if (lowerSrc.includes(pattern) || lowerClass.includes(pattern)) {
        score += 5;
      }
    }

    // Medium value indicators
    const mediumValuePatterns = [
      'product', 'item', 'merchandise', 'gallery', 'detail'
    ];
    for (const pattern of mediumValuePatterns) {
      if (lowerSrc.includes(pattern) || lowerAlt.includes(pattern) || lowerClass.includes(pattern)) {
        score += 3;
      }
    }

    // Low value indicators
    if (lowerSrc.includes('upload') || lowerSrc.includes('media') || lowerSrc.includes('cdn')) {
      score += 1;
    }

    // Negative indicators
    const negativePatterns = [
      'logo', 'icon', 'banner', 'header', 'footer', 'nav', 'menu',
      'social', 'facebook', 'twitter', 'instagram', 'youtube',
      'pixel', 'tracking', 'analytics', 'ads', 'advertisement',
      'sprite', 'background', 'bg-', 'thumbnail', 'thumb'
    ];
    for (const pattern of negativePatterns) {
      if (lowerSrc.includes(pattern) || lowerAlt.includes(pattern) || lowerClass.includes(pattern)) {
        score -= 3;
      }
    }

    // Size indicators
    const dimensionMatch = src.match(/(\d+)x(\d+)/);
    if (dimensionMatch) {
      const width = parseInt(dimensionMatch[1]);
      const height = parseInt(dimensionMatch[2]);
      if (width >= 400 && height >= 300) {
        score += 2;
      } else if (width >= 200 && height >= 200) {
        score += 1;
      } else if (width < 100 || height < 100) {
        score -= 2;
      }
    }

    return Math.max(0, score);
  }

  private static isProductImage(src: string, alt: string): boolean {
    return ImageService.scoreImage(src, alt, '') > 0;
  }

  private static resolveImageUrl(src: string, baseUrl: string): string {
    if (src.startsWith('http://') || src.startsWith('https://')) {
      return src;
    }
    
    if (src.startsWith('//')) {
      return `https:${src}`;
    }
    
    if (src.startsWith('/')) {
      const urlObj = new URL(baseUrl);
      return `${urlObj.protocol}//${urlObj.host}${src}`;
    }
    
    return new URL(src, baseUrl).href;
  }

  static async batchExtractImages(
    urls: string[], 
    concurrency: number = 5
  ): Promise<Map<string, ImageExtractionResult>> {
    const results = new Map<string, ImageExtractionResult>();
    for (let i = 0; i < urls.length; i += concurrency) {
      const batch = urls.slice(i, i + concurrency);
      const promises = batch.map(async (url) => {
        const result = await this.extractImageFromUrl(url);
        return { url, result };
      });
      const batchResults = await Promise.all(promises);
      for (const { url, result } of batchResults) {
        results.set(url, result);
      }
      // No delay between batches
    }
    return results;
  }
}