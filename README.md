# ArXiv Research Assistant

An advanced AI-powered research discovery platform that leverages cutting-edge machine learning technologies to provide intelligent and personalized academic research recommendations.

## Features
- Firebase authentication for secure access
- Real-time paper recommendations using GPT-4o
- Interactive paper discovery interface
- Research metrics and analytics
- Export functionality for saved papers

## Tech Stack
- Next.js with React for responsive frontend
- Firebase Realtime Database for data persistence
- Firebase authentication
- OpenAI GPT-4o for intelligent research analysis
- Advanced machine learning recommendation algorithms
- React Force Graph for interactive data visualization
- Vercel deployment with serverless architecture
- TypeScript for type-safe development

## Firebase Setup

1. Create a Firebase project:
   - Go to the [Firebase Console](https://console.firebase.google.com/)
   - Click "Add project" and follow the setup wizard
   - Enable Google Authentication in Authentication > Sign-in methods

2. Get your Firebase credentials:
   - Go to Project Settings > General
   - Scroll down to "Your apps" and click the web icon (</>)
   - Register your app and note down:
     - apiKey
     - projectId
     - appId

3. Set up Firebase Admin SDK:
   - Go to Project Settings > Service Accounts
   - Click "Generate New Private Key"
   - This will download a JSON file containing your admin credentials

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arxiv-research-assistant.git
cd arxiv-research-assistant
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file with the following variables:
```
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_APP_ID=your_app_id
FIREBASE_PRIVATE_KEY=your_private_key
FIREBASE_CLIENT_EMAIL=your_client_email
OPENAI_API_KEY=your_openai_key
```

4. Add your development URL to Firebase:
   - Go to Authentication > Settings in Firebase Console
   - Add your development URL to Authorized Domains

5. Start the development server:
```bash
npm run dev
```

## Deployment Instructions

1. Fork this repository to your GitHub account
2. Visit [Vercel](https://vercel.com/new) and click "Import Project"
3. Choose "Import Git Repository" and select this repository
4. Add the following environment variables in your Vercel project settings:
   - FIREBASE_API_KEY
   - FIREBASE_PROJECT_ID
   - FIREBASE_APP_ID
   - FIREBASE_PRIVATE_KEY
   - FIREBASE_CLIENT_EMAIL
   - OPENAI_API_KEY
5. Add your deployment URL to Firebase Authorized Domains
6. Click "Deploy"

Your application will be automatically built and deployed!

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.