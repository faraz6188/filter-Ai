const fs = require('fs');
const path = require('path');

const files = [
  path.join(__dirname, '../attached_assets/product_data_1750678538573.json'),
  path.join(__dirname, '../attached_assets/product_data_1750678073988.json'),
];

const categories = new Set();
const brands = new Set();
const prices = [];

for (const file of files) {
  if (!fs.existsSync(file)) continue;
  const raw = fs.readFileSync(file, 'utf8');
  let data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    console.error('Failed to parse', file);
    continue;
  }
  for (const item of data) {
    if (item.category && item.category !== 'nan') categories.add(item.category.trim());
    if (item.brand && item.brand !== 'nan') brands.add(item.brand.trim());
    // Try to extract price from original_text or price field
    let price = undefined;
    if (item.price && !isNaN(Number(item.price))) price = Number(item.price);
    else if (item.original_text) {
      const match = item.original_text.match(/$?(\d+(?:,\d+)*(?:\.\d{2})?)/);
      if (match) price = parseFloat(match[1].replace(/,/g, ''));
    }
    if (price) prices.push(price);
  }
}

console.log('Unique Categories:', Array.from(categories).sort());
console.log('Unique Brands:', Array.from(brands).sort());

// Price bands
prices.sort((a, b) => a - b);
const bands = [0, 225, 500, 1000, 2000, 5000, 10000];
const priceBands = bands.map((min, i) => {
  const max = bands[i + 1] || Infinity;
  const count = prices.filter(p => p >= min && (max === Infinity ? true : p < max)).length;
  return { min, max, count };
});
console.log('Price Bands:', priceBands); 