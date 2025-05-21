import axios from 'axios';
import { parseStringPromise } from 'xml2js';

export async function testArxivDeepLearningQuery() {
  const searchQuery = 'ti:"deep learning" OR abs:"deep learning"'; // Standard query for title or abstract
  const arxivApiUrl = 'http://export.arxiv.org/api/query';
  const params = {
    search_query: searchQuery,
    start: 0,
    max_results: 10, // Let's get a few results for testing
    sortBy: 'submittedDate', // Or 'relevance' or 'lastUpdatedDate'
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

    console.log('\n----- ArXiv API Response Headers -----');
    console.log(response.headers);

    console.log('\n----- ArXiv API Response Data (Raw XML) -----');
    console.log(response.data);

    // Optional: Parse and log number of entries
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

  } catch (error: any) {
    console.error('\n========== ERROR IN ARXIV DIRECT TEST ==========');
    console.error('Error fetching data from ArXiv:', error.message);
    if (error.response) {
      console.error('Error Response Status:', error.response.status);
      console.error('Error Response Data:', error.response.data);
    }
  }
  console.log('\n========== ARXIV DIRECT TEST FINISHED ==========');
}

// To run this test, you can call testArxivDeepLearningQuery() from your main server file (e.g., index.ts)
// or create a separate script to execute it.
// For example, add this line at the end of this file and run `node server/services/arxivTest.js` after compiling:
// testArxivDeepLearningQuery(); 