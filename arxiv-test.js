import axios from 'axios';
import { parseStringPromise } from 'xml2js';

async function testArxivQuery() {
  const searchQuery = 'ti:"deep learning" OR abs:"deep learning"'; // More standard format
  const arxivApiUrl = 'http://export.arxiv.org/api/query';
  const params = {
    search_query: searchQuery,
    start: 0,
    max_results: 10,
    sortBy: 'submittedDate',
    sortOrder: 'descending',
  };

  console.log('\n========== ARXIV DIRECT TEST STARTING ==========');
  console.log('Test Search Query:', searchQuery);
  console.log('API URL:', arxivApiUrl);
  console.log('Parameters:', params);

  try {
    const response = await axios.get(arxivApiUrl, { params });
    console.log('\n----- ArXiv API Response Status -----');
    console.log(response.status);
    console.log('\n----- ArXiv API Response Data (Raw XML) -----');
    console.log(response.data);

    const result = await parseStringPromise(response.data, {
      explicitArray: false,
      mergeAttrs: true,
    });

    const totalResults = result.feed['opensearch:totalResults'];
    const entries = result.feed.entry
      ? Array.isArray(result.feed.entry)
        ? result.feed.entry
        : [result.feed.entry]
      : [];
    
    console.log('\n----- Parsed Results Summary -----');
    console.log('Total Results reported by ArXiv:', totalResults);
    console.log('Number of entries in this response:', entries.length);

    if (entries.length > 0) {
      console.log('\nFirst entry title:', entries[0].title);
    }
  } catch (error) {
    console.error('\n========== ERROR IN ARXIV DIRECT TEST ==========');
    console.error('Error fetching data from ArXiv:', error.message);
    if (error.response) {
      console.error('Error Response Status:', error.response.status);
      console.error('Error Response Data:', error.response.data);
    }
  }
}

// Test "mech interp" query
async function testMechInterpQuery() {
  console.log('\n\n========== TESTING MECH INTERP QUERY ==========');
  // Test both formats to see which works
  const searchQueries = [
    '(title:"mech interp" OR abs:"mech interp")',
    'ti:"mech interp" OR abs:"mech interp"',
    'ti:mech interp OR abs:mech interp',
    'ti:"mechanistic interpretation" OR abs:"mechanistic interpretation"',
    'ti:"mechanistic interpretability" OR abs:"mechanistic interpretability"',
    'all:"mech interp" OR all:"mechanistic interp"'
  ];
  
  for (const query of searchQueries) {
    console.log(`\n----- Testing query: ${query} -----`);
    const arxivApiUrl = 'http://export.arxiv.org/api/query';
    const params = {
      search_query: query,
      start: 0,
      max_results: 5,
      sortBy: 'submittedDate',
      sortOrder: 'descending',
    };
    
    try {
      const response = await axios.get(arxivApiUrl, { params });
      console.log('Response Status:', response.status);
      
      const result = await parseStringPromise(response.data, {
        explicitArray: false,
        mergeAttrs: true,
      });
      
      const totalResults = result.feed['opensearch:totalResults'];
      const entries = result.feed.entry
        ? Array.isArray(result.feed.entry)
          ? result.feed.entry
          : [result.feed.entry]
        : [];
      
      console.log('Total Results:', totalResults);
      console.log('Number of entries:', entries.length);
    } catch (error) {
      console.error('Error:', error.message);
    }
  }
}

// Run the tests
await testArxivQuery();
await testMechInterpQuery();