import axios from 'axios';
import { parseStringPromise } from 'xml2js';

// Simulate the issue and fix
async function testSearch() {
  const originalQuery = '(title:"mech interp" OR abs:"mech interp")';
  const fixedQuery = originalQuery.replace(/title:/g, 'ti:').replace(/["']/g, '');
  
  console.log('Original query:', originalQuery);
  console.log('Fixed query:', fixedQuery);
  
  const arxivApiUrl = 'http://export.arxiv.org/api/query';
  
  // Test fixed query
  try {
    const response = await axios.get(arxivApiUrl, { 
      params: {
        search_query: fixedQuery,
        start: 0,
        max_results: 5
      }
    });
    
    const result = await parseStringPromise(response.data, {
      explicitArray: false,
      mergeAttrs: true
    });
    
    const totalResults = result.feed['opensearch:totalResults'];
    console.log('Total results with fixed query:', totalResults);
    
    const entries = result.feed.entry
      ? Array.isArray(result.feed.entry)
        ? result.feed.entry
        : [result.feed.entry]
      : [];
    
    console.log('Number of entries found:', entries.length);
    
    if (entries.length > 0) {
      console.log('\nExample papers found:');
      entries.slice(0, 3).forEach((entry, i) => {
        console.log(`\n${i+1}. Title: ${entry.title}`);
        console.log(`   Authors: ${Array.isArray(entry.author) ? entry.author.map(a => a.name).join(', ') : entry.author?.name || 'Unknown'}`);
        console.log(`   Abstract: ${entry.summary?.substring(0, 150)}...`);
      });
    }
  } catch (error) {
    console.error('Error:', error.message);
  }
}

testSearch();