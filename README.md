# ArXiv Research Assistant

An advanced AI-powered research discovery platform that leverages cutting-edge machine learning technologies to provide intelligent and personalized academic research recommendations.

## Features
- Firebase authentication for secure access
- Real-time paper recommendations using GPT-4
- Interactive paper discovery interface
- Research metrics and analytics
- Export functionality for saved papers

## Tech Stack
- Next.js with React for responsive frontend
- PostgreSQL for robust data persistence
- Firebase authentication
- OpenAI GPT-4o for intelligent research analysis
- Advanced machine learning recommendation algorithms
- React Force Graph for interactive data visualization
- Vercel deployment with serverless architecture
- TypeScript for type-safe development

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arxiv-research-assistant.git
cd arxiv-research-assistant
```

2. Install dependencies:
```bash
npm install
cd client && npm install
```

3. Create a `.env` file with the following variables:
```
DATABASE_URL=your_postgres_url
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_APP_ID=your_app_id
FIREBASE_PRIVATE_KEY=your_private_key
FIREBASE_CLIENT_EMAIL=your_client_email
OPENAI_API_KEY=your_openai_key
```

4. Start the development server:
```bash
npm run dev
```

## Deployment Instructions

1. Fork this repository to your GitHub account
2. Visit [Vercel](https://vercel.com/new) and click "Import Project"
3. Choose "Import Git Repository" and select this repository
4. Add the following environment variables in your Vercel project settings:
   - DATABASE_URL
   - FIREBASE_API_KEY
   - FIREBASE_PROJECT_ID
   - FIREBASE_APP_ID
   - FIREBASE_PRIVATE_KEY
   - FIREBASE_CLIENT_EMAIL
   - OPENAI_API_KEY
5. Click "Deploy"

Your application will be automatically built and deployed!

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.