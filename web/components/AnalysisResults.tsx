import { motion } from 'framer-motion';

interface AnalysisResultsProps {
  results: {
    personality: string;
    recommendations: string[];
    user_rating: number;
  };
}

export default function AnalysisResults({ results }: AnalysisResultsProps) {
  const { personality, recommendations, user_rating } = results;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 bg-white rounded-lg shadow-lg"
    >
      <h2 className="text-2xl font-bold mb-4">Your Financial Analysis</h2>
      
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Personality Type</h3>
        <p className="text-lg text-blue-600">{personality}</p>
      </div>

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Recommendations</h3>
        <ul className="space-y-2">
          {recommendations.map((rec, index) => (
            <motion.li
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-3 bg-gray-50 rounded"
            >
              {rec}
            </motion.li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="text-xl font-semibold mb-2">Financial Health Score</h3>
        <div className="relative pt-1">
          <div className="flex mb-2 items-center justify-between">
            <div className="text-lg font-semibold">{user_rating}/10</div>
          </div>
          <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(user_rating/10)*100}%` }}
              className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
