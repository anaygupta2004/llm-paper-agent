import axios from 'axios';
import { parseStringPromise } from 'xml2js';

async function testQuery() {
  // Test the problematic search query but with the fixed format
  const searchQuery = '(ti:"mech interp" OR abs:"mech interp")';
  const fallbackQuery = 'ti:mech interp OR abs:mech interp'; // Without quotes
  
  console.log('Testing search queries with proper ArXiv API format');
  
  const arxivApiUrl = 'http://export.arxiv.org/api/query';
  
  // Test first query
  console.log('\nQuery 1:', searchQuery);
  try {
    const response1 = await axios.get(arxivApiUrl, { 
      params: {
        search_query: searchQuery,
        start: 0,
        max_results: 5
      }
    });
    
    const result1 = await parseStringPromise(response1.data, {
      explicitArray: false,
      mergeAttrs: true
    });
    
    const totalResults1 = result1.feed['opensearch:totalResults'];
    console.log('Results for fixed format with quotes:', totalResults1);
  } catch (error) {
    console.error('Error with query 1:', error.message);
  }
  
  // Test fallback query
  console.log('\nQuery 2:', fallbackQuery);
  try {
    const response2 = await axios.get(arxivApiUrl, { 
      params: {
        search_query: fallbackQuery,
        start: 0,
        max_results: 5
      }
    });
    
    const result2 = await parseStringPromise(response2.data, {
      explicitArray: false,
      mergeAttrs: true
    });
    
    const totalResults2 = result2.feed['opensearch:totalResults'];
    console.log('Results for format without quotes:', totalResults2);
    
    if (totalResults2 > 0) {
      const entries = result2.feed.entry
        ? Array.isArray(result2.feed.entry)
          ? result2.feed.entry
          : [result2.feed.entry]
        : [];
      
      console.log('Number of entries:', entries.length);
      if (entries.length > 0) {
        console.log('First entry title:', entries[0].title);
      }
    }
  } catch (error) {
    console.error('Error with query 2:', error.message);
  }
}

// Run the test
testQuery();