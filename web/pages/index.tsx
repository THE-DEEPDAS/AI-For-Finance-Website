import { useState } from 'react';
import Head from 'next/head';
import UploadSection from '../components/UploadSection';
import AnalysisResults from '../components/AnalysisResults';
import Header from '../components/Header';

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalysis = async (data) => {
    setLoading(true);
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: data }),
      });
      const result = await response.json();
      setAnalysisResult(result);
    } catch (error) {
      console.error('Analysis failed:', error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-100 to-white">
      <Head>
        <title>Financial Personality Analyzer</title>
        <meta name="description" content="AI-powered financial analysis and recommendations" />
      </Head>

      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <UploadSection onAnalyze={handleAnalysis} />
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin h-12 w-12 border-4 border-blue-500 rounded-full border-t-transparent"></div>
            </div>
          ) : (
            analysisResult && <AnalysisResults results={analysisResult} />
          )}
        </div>
      </main>
    </div>
  );
}
